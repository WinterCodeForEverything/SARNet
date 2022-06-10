import logging
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import device
import torchvision
from tensorboardX import SummaryWriter

#ArgumentParser
from arguments import train_arguments

#dataloader
from dataloader.transforms import get_transforms
from dataloader.NuScenesDataLoader import NuScenesDataSet
from dataloader.SemanticKITTYDataLoader import KittiDataset
from torch.utils.data.distributed import DistributedSampler

#loss 
from models.loss import Loss

#common
from common.torch import CheckPointManager, dict_all_to_device, to_numpy
from common.misc import prepare_logger

#metrics
from metrics import compute_metrics, summarize_metrics, print_metrics

#others
from tqdm import tqdm
from typing import Dict
from collections import defaultdict


parser = train_arguments()
_args = parser.parse_args()
#initialize
torch.distributed.init_process_group(backend="nccl", world_size=2)
#get gpu
_local_rank = torch.distributed.get_rank()
torch.cuda.set_device(_local_rank)
_device = torch.device("cuda", _local_rank)

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce( rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def save_summaries( writer: SummaryWriter, losses: Dict = None, metrics: Dict = None, step: int = 0):
    """Save tensorboard summaries"""
    with torch.no_grad():
        if losses is not None:
            for l in losses:
                writer.add_scalar( 'losses/{}'.format(l), losses[l], step )
        if metrics is not None:
            for m in metrics:
                writer.add_scalar( 'metrics/{}'.format(m), metrics[m], step )
        writer.flush()


def validate( data_loader, criteria: nn.Module, writer: SummaryWriter, step: int = 0 ):
    """Perform a single validation run"""

    with torch.no_grad():
        val_losses_np = defaultdict(list)
        val_metrics_np = defaultdict(list)
        for data in data_loader:
            dict_all_to_device( data, _device )
            predict, losses = criteria( data )
            metrics = compute_metrics( data, predict['pred_transforms'][-1] )
            for k in metrics:
                val_metrics_np[k].append( metrics[k] )
            for k in losses:
                val_losses_np[k].append( to_numpy(losses[k]) )
        val_losses_np =  { k : np.mean( val_losses_np[k] ) for k in val_losses_np }
        val_metrics_np = { k : np.concatenate( val_metrics_np[k] ) for k in val_metrics_np }
        summary_metrics = summarize_metrics( val_metrics_np, _args.RRE_thresholds, _args.RTE_thresholds )    
        print_metrics( _logger, summary_metrics )

        score = -val_losses_np['trans']

        save_summaries( writer, val_losses_np, summary_metrics, step )
        return score

def main():
    #dataloader
    train_loader, val_loader = None, None
    train_transform, val_trainsform = get_transforms( noise_type = _args.noise_type, 
        rot_mag = _args.rot_mag, trans_mag = _args.trans_mag, voxel_size= _args.sample_voxel_size, 
        num = _args.sample_point_num, diagonal=  _args.boundingbox_diagonal, partial_p_keep = _args.partial_p_keep
    )
    train_transform = torchvision.transforms.Compose( train_transform )
    val_trainsform = torchvision.transforms.Compose( val_trainsform )
    if _args.dataset == 'NuScenes':
        train_set = NuScenesDataSet( root = _args.nuscenes_root, split='train', 
            transform = train_transform, ignore_label= _args.nuscenes_ignore_label, augment= _args.augment )
        train_loader = torch.utils.data.DataLoader( train_set, batch_size=_args.train_batch_size, num_workers= _args.num_workers, sampler=DistributedSampler(train_set) )

        val_set = NuScenesDataSet( root = _args.nuscenes_root, split='val', 
            transform = val_trainsform, ignore_label= _args.nuscenes_ignore_label, augment= _args.augment )
        val_loader = torch.utils.data.DataLoader( val_set, batch_size=_args.val_batch_size, num_workers=_args.num_workers )        
    elif _args.dataset == 'SemanticKitti':
        train_set = KittiDataset( root = _args.kitty_root, split='train',
            transform = train_transform, ignore_label= _args.kitti_ignore_label, augment= _args.augment )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=_args.train_batch_size, num_workers= _args.num_workers, sampler=DistributedSampler(train_set) ) 

        val_set = KittiDataset( root = _args.kitty_root, split='val',  
            transform = val_trainsform, ignore_label= _args.kitti_ignore_label, augment= _args.augment
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=_args.train_batch_size, num_workers= _args.num_workers)  #, sampler=DistributedSampler(val_set)


    #SummaryWriter
    if _local_rank == 0:
        train_writer = SummaryWriter(osp.join(_log_path, 'train'), flush_secs=60)
        val_writer = SummaryWriter(osp.join(_log_path, 'val'), flush_secs=60)

    #model
    criteria = Loss( _args )
    criteria.to( _device )

    if torch.cuda.device_count() > 1:
        criteria = torch.nn.parallel.DistributedDataParallel(criteria,
                                                        device_ids=[_local_rank],
                                                        output_device=_local_rank,
                                                        find_unused_parameters=True)

    #optimizer
    optimizer = torch.optim.Adam( criteria.parameters(), lr= _args.lr )
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=_args.scheduler_step_size, gamma=_args.scheduler_gamma )

    #checkpoints
    global_step = 0
    saver = CheckPointManager( _args.save_checkpoints_path, max_to_keep = 1, keep_checkpoint_every_n_hours = 0.1 )

    if osp.exists( _args.load_checkpoints_path ):
        global_step = saver.load( _args.load_checkpoints_path, criteria, optimizer )

    if _local_rank == 0:        
        steps_per_epoch = len(train_loader)
        if _args.validate_every < 0:
            _args.validate_every = abs(_args.validate_every) * steps_per_epoch
        if _args.summary_every < 0:
            _args.summary_every = abs(_args.summary_every) * steps_per_epoch

    #model training
    criteria.train()
    for epoch in range(_args.epoch_num):
        if _local_rank == 0:
            tbar = tqdm(total=len(train_loader), ncols=100)
        for data in train_loader:  
            optimizer.zero_grad()
            dict_all_to_device( data, _device )
            
            _, losses = criteria( data )
            losses['total'].backward()
            if validate_gradient( criteria ):
                optimizer.step()
            else:
                print("gradient not valid")

            global_step += 1
            avg_total_loss = reduce_tensor(losses['total']).item()
            if _local_rank == 0:
                tbar.set_description('Epoch:{:.4g} Loss:{:.4g}'.format(epoch, avg_total_loss ))
                tbar.update(1)

                if global_step % _args.validate_every == 0:
                    criteria.eval()
                    score = validate( val_loader, criteria, val_writer, global_step )
                    saver.save( criteria, optimizer, global_step, score = score )
                    criteria.train()
                if global_step % _args.summary_every == 0:
                    save_summaries( train_writer, losses, step = global_step )
        scheduler.step()
        if _local_rank == 0:
            tbar.close()

if __name__ == '__main__':
    if _local_rank == 0:
        _logger, _log_path = prepare_logger(_args)
    main()

