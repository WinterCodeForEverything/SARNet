import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(data, k=20):
    xyz = data
    # x = x.squeeze()
    idx = knn(xyz, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = xyz.size()

    xyz = xyz.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)

    # gxyz
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)
    # xyz
    xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #lxyz_norm
    neighbor_lxyz_norm = torch.norm(neighbor_gxyz - xyz, dim=3, keepdim=True)

    feature = torch.cat((xyz, neighbor_gxyz, neighbor_lxyz_norm), dim=3)

    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature


class SemanticConv(nn.Module):
    def __init__(self, in_dim, out_dim, neighboursnum=16):
        super(SemanticConv, self).__init__()
        # 确定输入的点云信息
        self.neighboursnum = neighboursnum
        self.in_dim = in_dim

        self.localConv = nn.Sequential(
            nn.Conv2d(in_dim*2+1, out_dim, kernel_size=1, bias=False ),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )
        self.semConv =  nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False ),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
        self.fullConv = nn.Sequential(
            nn.Conv1d(out_dim*2, out_dim, kernel_size=1, bias=False ),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
        self.semAtt =  nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.semAtt) for _ in range(3)])
        #self.conv_a = nn.Conv1d( in_dim, in_dim, kernel_size=1 )


    def forward( self, f_in ):    # f_in:(B, C, N)
        neighbor_f_in = get_graph_feature(f_in, self.neighboursnum)   # (B, C, N, n)
        Intra_channal = self.localConv( neighbor_f_in )
        Intra_channal = Intra_channal.max(dim=-1, keepdim=False)[0]
        q, k, v = [ l(f_in) for l in self.proj ]
        scores = torch.einsum('bdm,bdn->bmn', q, k) / self.in_dim**.5
        scores = torch.softmax(scores, dim=-1)
        fgt = torch.einsum('bmn,bdn->bdm', scores, v)
        #a = (self.conv_a( fgt ) + neighbor_mean)/2
        Inter_channal = self.semConv(fgt)
        feature = self.fullConv( torch.cat( (Intra_channal, Inter_channal), dim= 1 ) )
        return feature      #, Inter_channal


class SemanticCNN(nn.Module):
    def __init__(self, raw_dim, emb_dim, neighboursnum=16):
        super(SemanticCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d( raw_dim, emb_dim//16, kernel_size=1 ),
            nn.BatchNorm1d(emb_dim//16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d( emb_dim, emb_dim, kernel_size=1 ),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Conv1d( emb_dim, emb_dim, kernel_size=1 )
        )
        #self.conv3 = nn.Sequential(
        #    nn.Conv1d( emb_dim*2, emb_dim, kernel_size=1 ),
        #    nn.BatchNorm1d(emb_dim),
        #    nn.ReLU(),
        #    nn.Conv1d( emb_dim, emb_dim, kernel_size=1 )
        #)
        self.sem1 = SemanticConv( emb_dim//16, emb_dim//16, neighboursnum )
        self.sem2 = SemanticConv(  emb_dim//16, emb_dim//8, neighboursnum )
        self.sem3 = SemanticConv(  emb_dim//8, emb_dim//4, neighboursnum )
        self.sem4 = SemanticConv( emb_dim//4, emb_dim//2, neighboursnum )

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1).contiguous()     #(B, 3, N)
        #points_num = xyz.shape[2]
        x0 = self.conv1( xyz )
        x1 = self.sem1(x0)
        x2 = self.sem2(x1)
        x3 = self.sem3(x2)
        x4 = self.sem4(x3)

        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        #gx = torch.cat((x0, gx1, gx2, gx3, gx4), dim=1)
        #gx_m = torch.max( gx, dim=2, keepdim=True )[0]
        #gx_f = torch.cat((gx_m.repeat( 1, 1, points_num), gx), dim = 1)
        feature = self.conv2( x )
        #semantic = self.conv3( gx_f )
        return feature
