from collections import defaultdict
import os
import os.path as osp
import numpy as np
import json
import pandas as pd
import time

import torch
import torchvision
import torch.utils.data


#ArgumentParser
from arguments import test_arguments

#dataloader
from dataloader.transforms import get_transforms
from dataloader.NuScenesDataLoader import NuScenesDataSet
from dataloader.SemanticKITTYDataLoader import KittiDataset

#model
from models.loss import Loss

#metrics
from metrics import compute_metrics, summarize_metrics, print_metrics

#common
from tqdm import tqdm
from common.torch import CheckPointManager, dict_all_to_device, to_numpy
from common.math_torch import se3
from common.misc import prepare_logger

parser = test_arguments()
_args = parser.parse_args()
_device = torch.device("cuda:0")


def test( pred_transforms, data_loader: torch.utils.data.dataloader.DataLoader):
    """ Test the computed transforms against the groundtruth

    Args:
        pred_transforms: Predicted transforms (N, B, 3/4, 4)
        data_loader: Loader for dataset.

    Returns:
        Computed metrics (List of dicts), and summary metrics (only for last iter)
    """

    _logger.info('Testing transforms...')
    all_metrics_np = defaultdict(list)

    num_processed = 0
    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, _device)
        batch_size = data['points_src'].shape[0]
        metrics = compute_metrics(data, pred_transforms[num_processed:num_processed+batch_size, :, :] )
        num_processed += batch_size
        for k in metrics:
            all_metrics_np[k].append( metrics[k] )
    all_metrics_np = {k: np.concatenate(all_metrics_np[k]) for k in all_metrics_np}
    summary_metrics = summarize_metrics(all_metrics_np, _args.RRE_thresholds, _args.RTE_thresholds )
    print_metrics(_logger, summary_metrics, title='Evaluation result')

    return all_metrics_np, summary_metrics


def inference(data_loader, model: torch.nn.Module):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    Returns:
        pred_transforms_all: predicted transforms (N, B, 3, 4) where N is total number of instances
        endpoints_out (Dict): Network endpoints
    """

    _logger.info('Starting inference...')
    model.eval()

    pred_transforms_all = []
    total_time = 0.0
    total_rotation = []

    with torch.no_grad():
        
        for test_data in tqdm(data_loader):
            rot_trace = test_data['transform_gt'][:, 0, 0] + test_data['transform_gt'][:, 1, 1] + \
                        test_data['transform_gt'][:, 2, 2]
            rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
            total_rotation.append(np.abs(to_numpy(rotdeg)))

            dict_all_to_device(test_data, _device)
            time_before = time.time()
            pred = model( test_data['points_src'], test_data['points_ref'],
                 test_data['seg_src'], test_data['seg_ref'],
                 test_data['intersect_src'], test_data['intersect_ref'] ) 
            pred_transforms = pred['pred_transforms']
            total_time += time.time() - time_before
            pred_transforms_all.append(to_numpy(pred_transforms[-1]))

    _logger.info('Total inference time: {}s'.format(total_time))
    total_rotation = np.concatenate(total_rotation, axis=0)
    _logger.info('Rotation range in data: {}(avg), {}(max)'.format(np.mean(total_rotation), np.max(total_rotation)))
    pred_transforms_all = np.concatenate(pred_transforms_all, axis=0)

    return pred_transforms_all

def save_eval_data(pred_transforms, metrics, summary_metrics, save_path):
    """Saves out the computed transforms
    """

    # Save transforms
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    # Save metrics
    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    metrics.pop('err_r_deg')   
    metrics.pop('err_t')
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.to_excel(writer, sheet_name='metrics')
    writer.close()

    # Save summary metrics
    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as json_out:
        json.dump(summary_metrics_float, json_out)

    _logger.info('Saved evaluation results to {}'.format(save_path))

def get_model():
    criteria = Loss( _args )
    criteria.to( _device )
    save_path = os.path.join( _args.checkpoints_path, 'ckpt')
    saver = CheckPointManager( save_path )
    load_path = os.path.join( _args.checkpoints_path, 'ckpt-best.pth' )
    global_step = 0
    if os.path.exists(load_path):
        global_step = saver.load( load_path, criteria, distributed=True )
    print( "global_step:", global_step )
    model = criteria.model
    model.eval()
    return model


def main():
    #dataloader
    test_set, test_loader = None, None
    _, test_trainsform = get_transforms( noise_type = _args.noise_type, 
        rot_mag = _args.rot_mag, trans_mag = _args.trans_mag, voxel_size= _args.sample_voxel_size,
        num = _args.sample_point_num, diagonal=  _args.boundingbox_diagonal, partial_p_keep = _args.partial_p_keep
    )
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_trainsform])))
    test_trainsform = torchvision.transforms.Compose(test_trainsform)
    if _args.dataset == 'NuScenes':
        test_set = NuScenesDataSet( root = _args.nuscenes_root, split='test',
             transform = test_trainsform, ignore_label= _args.nuscenes_ignore_label, augment= _args.augment )
        test_loader = torch.utils.data.DataLoader( test_set, batch_size=_args.test_batch_size, shuffle=False, num_workers=_args.num_workers)   
    elif _args.dataset == 'SemanticKitti':
        test_set = KittiDataset( root = _args.kitty_root, split='test',
                 transform = test_trainsform, ignore_label= _args.kitti_ignore_label, augment= _args.augment ) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=_args.test_batch_size, shuffle=False, num_workers= _args.num_workers)

    #model
    if _args.transform_file is not None:
        _logger.info('Loading from precomputed transforms: {}'.format(_args.transform_file))
        pred_transforms = np.load(_args.transform_file)
    else:
        model = get_model()
        pred_transforms = inference(test_loader, model) # Feedforward transforms

    eval_metrics, summary_metrics = test( torch.from_numpy(pred_transforms).to(_device), data_loader=test_loader )
    save_eval_data( pred_transforms, eval_metrics, summary_metrics, _args.eval_save_path )

if __name__ == '__main__':
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path )
    #_args.transform_file = osp.join( _args.eval_save_path, 'pred_transforms.npy')
    main()
