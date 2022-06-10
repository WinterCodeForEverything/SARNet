import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils import furthest_point_sample, weighted_furthest_point_sample, gather_operation


class KeypointDetector(nn.Module):
    def __init__(self, nsample, sem_num, sample_type = 'fps'):
        super(KeypointDetector, self).__init__()
        self.nsample = nsample
        self.sample_type = sample_type
        self.semantic_classes_num = sem_num


    def forward( self, xyz, seg_feature, seg_label = None, weights = None ): 
        # Use FPS or random sampling
        B, N, C = xyz.shape
        sample_seg_feature, sample_seg_label = None, None
        seg_weights = None
        if seg_label != None:
            sem_one_hot = F.one_hot( seg_label.long(), num_classes=self.semantic_classes_num )
            count = torch.sum( sem_one_hot, dim= 1 )
            seg_weights = torch.gather( count, dim= 1, index= seg_label.long() ).float()
            if weights != None:
                assert( seg_weights.shape == weights.shape )
                seg_weights = seg_weights * weights
            if self.sample_type == 'fps':
                # Use WFPS
                idx = weighted_furthest_point_sample(xyz, seg_weights, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), idx).permute(0,2,1).contiguous()
            else:
                idx = torch.multinomial( seg_weights, self.nsample )
                sampled_xyz = torch.gather( xyz, dim= 1, index= idx.unsqueeze(-1).repeat(1,1,C).long() )
        else:
            if self.sample_type == 'fps':
                idx = furthest_point_sample(xyz, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), idx).permute(0,2,1).contiguous()
            else:
                idx = torch.randperm(N)[:self.nsample]
                sampled_xyz = xyz[:,idx,:]
                idx = idx.unsqueeze(0).repeat(B, 1)
        sample_seg_feature = torch.gather( seg_feature, dim=-1, index=idx.unsqueeze(1).repeat(1, 
                seg_feature.shape[1], 1).long())
        if seg_label != None:
            sample_seg_label = torch.gather( seg_label, dim=1, index=idx.long())

        return sampled_xyz, sample_seg_feature, sample_seg_label

if __name__ == '__main__':
    _device = torch.device("cuda:0")
    xyz = Variable( torch.rand((2, 8, 3)))
    sem = Variable( torch.randint(0 , 20, (2, 8)) )
    detector = KeypointDetector( 4, 20, sample_type = 'fps' )
    sampled_xyz, idx = detector( xyz, sem )
