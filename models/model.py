import argparse
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from common.math_torch import se3

#from models.socnn import SOCNN
from models.RandLA_Net import RandLANet
from models.key_point_dectector import KeypointDetector
from models.semanticCNN import SemanticCNN
from models.attention import AttentionalPropagation
from models.compute_rigid_transform import weighted_svd

class PermutationWeights(nn.Module):
    def __init__( self, in_dim ):
        super( PermutationWeights, self ).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim*2, in_dim*2, kernel_size=1),
                                  nn.BatchNorm1d(in_dim*2),
                                  nn.ReLU(),
                                  nn.Conv1d(in_dim*2, in_dim, kernel_size=1),
                                  nn.BatchNorm1d(in_dim),
                                  nn.ReLU(),
                                  nn.Conv1d(in_dim, 1, kernel_size=1)
                                  )

    def forward( self, features_src: torch.Tensor, featrues_ref: torch.Tensor,
        part_match_matrix: torch.Tensor = None ):
        d_k = features_src.size(1)
        scores = torch.matmul( features_src.transpose(1,2), featrues_ref )/math.sqrt(d_k)

        if part_match_matrix != None:
            assert( scores.shape == part_match_matrix.shape )
            scores[part_match_matrix==0] = -1e10
        permutation = torch.softmax( scores, dim=-1 )
        featrues_ref_perm = torch.bmm( permutation, featrues_ref.transpose(1,2) ).transpose(1,2).contiguous()        
        weights = self.conv(torch.cat( (features_src, featrues_ref_perm), dim=1 ))
        weights_s = torch.sigmoid( weights.squeeze(1) )
        return permutation, weights_s

class SARNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super( SARNet, self ).__init__()
        self.iter_num = args.iter_num
        self.classes_num = args.semantic_classes_num
        self.segcnn = RandLANet(
                args.init_dims,
                args.semantic_classes_num,
                num_neighbors=args.nb
            )
        self.detector = KeypointDetector( args.nsample, args.semantic_classes_num )
        self.regcnn = nn.ModuleList()
        self.attention_fea= nn.ModuleList()
        self.semconv, self.conv = nn.ModuleList(), nn.ModuleList()
        self.perm_weights = nn.ModuleList()
        for i in range(self.iter_num):
            self.regcnn += [SemanticCNN( args.init_dims, args.emb_dims, args.nb )]
            self.attention_fea += [AttentionalPropagation( args.emb_dims, args.attention_head_num )]
        
            self.conv += [nn.Sequential(
                nn.Conv1d( args.emb_dims + args.semantic_classes_num, args.emb_dims, kernel_size=1 ), 
                nn.BatchNorm1d( args.emb_dims ),
                nn.ReLU(),
                nn.Conv1d( args.emb_dims, args.emb_dims, kernel_size=1 )
            )]   

            self.perm_weights += [PermutationWeights( args.emb_dims )]

    def forward( self, *input ):
        points_src, points_ref = input[0], input[1]
        seg_label_src, seg_label_ref = None, None
        weights_src, weights_ref = None, None
        if len(input) == 4:
            seg_label_src, seg_label_ref = input[2], input[3]
        if len(input) == 6:
            seg_label_src, seg_label_ref = input[2], input[3]
            weights_src, weights_ref = input[4], input[5]



        seg_src = self.segcnn( points_src )
        seg_ref = self.segcnn( points_ref )
        seg_src = torch.distributions.utils.probs_to_logits(seg_src, is_binary=False)
        seg_ref = torch.distributions.utils.probs_to_logits(seg_ref, is_binary=False)
        seg_src_detach = seg_src.detach()
        seg_ref_detach = seg_ref.detach()
        points_src, sample_seg_src, sample_seg_label_src = self.detector( 
            points_src, seg_src_detach, seg_label = seg_label_src, weights = weights_src )
        points_ref, sample_seg_ref, sample_seg_label_ref = self.detector( 
            points_ref, seg_ref_detach, seg_label = seg_label_ref, weights = weights_ref )
        onehot_src, onehot_ref = None, None
        if sample_seg_label_src != None and sample_seg_label_ref != None:
            onehot_src = F.one_hot( sample_seg_label_src.long(), num_classes = self.classes_num ).float()
            onehot_ref = F.one_hot( sample_seg_label_ref.long(), num_classes = self.classes_num ).float()
        else:
            onehot_src = F.one_hot( sample_seg_src.max(dim=1)[1].long(), num_classes = self.classes_num ).float()
            onehot_ref = F.one_hot( sample_seg_ref.max(dim=1)[1].long(), num_classes = self.classes_num ).float()

        part_match_matrix = torch.bmm( onehot_src, onehot_ref.transpose(1,2) ).int()
        onehot_src = onehot_src.transpose(1,2).contiguous()
        onehot_ref = onehot_ref.transpose(1,2).contiguous()

        points_src_t = points_src
        transforms = [None]*self.iter_num
        for i in range( self.iter_num ):
            features_src = self.regcnn[i]( points_src_t )
            features_ref = self.regcnn[i]( points_ref )
            features_src = features_src + self.attention_fea[i]( features_src, features_ref )
            features_ref = features_ref + self.attention_fea[i]( features_ref, features_src )

            features_seg_src = self.conv[i](torch.cat(( features_src, onehot_src ), dim=1 ))
            features_seg_ref = self.conv[i](torch.cat(( features_ref, onehot_ref ), dim=1 ))

            permutation, weights = self.perm_weights[i]( features_seg_src, features_seg_ref, part_match_matrix )
            transform = weighted_svd( points_src, points_ref, weights, permutation )
            points_src_t = se3.transform( transform.detach(), points_src_t )
            transforms[i] = transform

        predict = { 'pred_transforms': transforms,
            'seg_src': seg_src,
            'seg_ref': seg_ref,
            'seg_label_src': seg_label_src,
            'seg_label_ref': seg_label_ref
        }
        return predict

if __name__ == '__main__':
    pass



