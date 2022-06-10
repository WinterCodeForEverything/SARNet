import argparse
import torch
import torch.nn as nn

from typing import List
from common.math_torch import se3

#model
from models.model import SARNet

class Loss(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Loss, self).__init__()
        self.model = SARNet( args )
        self.trans_loss_type = args.trans_loss_type
        self.seg_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.reg_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

    def forward(self, data: dict ):
        losses = {}
        predict = self.model( data['points_src'], data['points_ref'],
                         data['seg_src'], data['seg_ref'],
                         data['intersect_src'], data['intersect_ref'] )
        losses_trans_iter = self.compute_trans_loss( data, predict['pred_transforms'], self.trans_loss_type )
        discount_factor = 0.5  # Early iterations will be discounted
        iter_num = len( losses_trans_iter )
        for i in range(iter_num):
            discount = discount_factor ** ( iter_num-i-1 )
            losses_trans_iter[i] *= discount
        losses['trans'] = torch.sum(torch.stack(losses_trans_iter))
        losses['semantic'] = self.compute_semantic_loss( predict )
        factor_reg = 1.0 / (self.reg_sigma**2)
        factor_seg = 1.0 / (self.seg_sigma**2)
        losses['total'] = factor_reg*losses['trans'] + factor_seg*losses['semantic'] + \
                2 * torch.log(self.reg_sigma) + 2 * torch.log(self.seg_sigma)
        return  predict, losses


    def compute_trans_loss(self, data: dict, pred_transforms: List,
            loss_type: str = 'mse', reduction: str= 'mean' ):
        # Compute losses
        losses = []
        iter_num = len( pred_transforms )
        gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])
        if loss_type == 'mse':
            # MSE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.MSELoss( reduction= reduction ) 
            for i in range( iter_num ):
                pred_src_transformed = se3.transform( pred_transforms[i], data['points_src'][..., :3] )
                losses.append(criterion(pred_src_transformed, gt_src_transformed))
        elif loss_type == 'mae':
            # MAE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.L1Loss( reduction= reduction )
            for i in range( iter_num ):
                pred_src_transformed = se3.transform( pred_transforms[i], data['points_src'][..., :3] )
                losses.append(criterion(pred_src_transformed, gt_src_transformed))
        else:
            raise NotImplementedError
        return  losses

    def compute_semantic_loss(self, predict: dict ):
        criterion = nn.CrossEntropyLoss()
        seg_src = predict['seg_src']
        seg_ref = predict['seg_ref']
        sem_label_src = predict['seg_label_src']
        sem_label_ref = predict['seg_label_ref']
        loss = criterion( seg_src, sem_label_src.long() ) + criterion( seg_ref, sem_label_ref.long() )
        return loss
