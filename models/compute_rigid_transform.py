import numpy as np
from sympy.matrices import Matrix, GramSchmidt
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn

_EPS = 1e-5  # To prevent division by zero

def svd(src: torch.Tensor, ref: torch.Tensor, permutation: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        src (torch.Tensor): (B, M, 3) points
        ref (torch.Tensor): (B, N, 3) points
        permutation (torch.Tensor): (B, M, N)

    Returns:
        Transform T (B, 3, 4) to get from src to ref, i.e. T*src = ref
    """

    ref_perm = torch.bmm( permutation, ref )
    center_src = torch.mean( src, dim=1, keepdim=True )
    center_ref = torch.mean( ref_perm, dim=1, keepdim=True )
    src_c = src - center_src
    ref_c = ref_perm - center_ref

    H = torch.bmm( src_c.transpose( 1, 2 ), ref_c )
    u, s, v = torch.svd( H, some=False )
    R_pos = torch.bmm( v, u.transpose(1, 2) )
    v_neg = v.clone()
    v_neg[:, :, 2] *=-1
    R_neg = torch.bmm( v_neg, u.transpose(1, 2) )
    R = torch.where( torch.det(R_pos)[:, None, None] > 0, R_pos, R_neg )
    assert torch.all(torch.det(R) > 0)

    T = center_ref.transpose(1,2) - torch.bmm( R, center_src.transpose(1,2) )
    transform = torch.cat((R, T), dim=2)
    return transform

def weighted_svd( src: torch.Tensor, ref: torch.Tensor, 
    weights: torch.Tensor, permutation: torch.Tensor = None ):
    sum_weights = torch.sum(weights,dim=1,keepdim=True) + _EPS
    weights = weights/sum_weights
    weights = weights.unsqueeze(2)

    ref_perm = ref
    if permutation != None:
        ref_perm = torch.bmm( permutation, ref_perm )
    
    src_mean = torch.matmul(weights.transpose(1,2),src)/(torch.sum(weights,dim=1).unsqueeze(1)+_EPS)
    src_corres_mean = torch.matmul(weights.transpose(1,2),ref_perm)/(torch.sum(weights,dim=1).unsqueeze(1)+_EPS)
    src_centered = src - src_mean # [B,N,3]
    src_corres_centered = ref_perm - src_corres_mean # [B,N,3]
    weight_matrix = torch.diag_embed(weights.squeeze(2))
    
    cov_mat = torch.matmul(src_centered.transpose(1,2),torch.matmul(weight_matrix,src_corres_centered))
    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3).cuda()
        r = r.repeat(src_mean.shape[0],1,1)
        t = torch.zeros((src_mean.shape[0],3,1)).cuda()
        #t = t.view(t.shape[0], 3)
        transform = torch.cat((r, t), dim=2)
        return transform
    
    tm_determinant = torch.det(torch.matmul(v.transpose(1,2), u.transpose(1,2)))
    
    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0], 2)).cuda(),tm_determinant.unsqueeze(1)), 1))
    r = torch.matmul(v, torch.matmul(determinant_matrix, u.transpose(1,2)))
    t = src_corres_mean.transpose(1,2) - torch.matmul(r, src_mean.transpose(1,2))
    #t = t.view(t.shape[0], 3)
    
    transform = torch.cat((r, t), dim=2)
    return transform

        
def orthogo_tensor(x):
    m, n = x.size()
    x_np = x.t().numpy()
    matrix = [Matrix(col) for col in x_np.T]
    gram = GramSchmidt(matrix)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))
        ort_list.append(vector)
    ort_list = np.mat(ort_list)
    ort_list = torch.from_numpy(ort_list)
    ort_list = F.normalize(ort_list,dim=1)
    return ort_list

if __name__ == '__main__':
    src = torch.randn((8, 128, 3))
    R_r = torch.randn((8,3,3))
    for i in range(8):
        R_r[i,:,:] = orthogo_tensor(R_r[i,:,:])
    T_r = torch.randn((8,3,1))
    ref = torch.bmm( R_r, src.transpose(1,2) ) + T_r
    ref = ref.transpose(1,2).contiguous()
    perm = torch.eye( 128 ).reshape((1, 128, 128)).repeat(8, 1, 1)
    transform = svd( src, ref, perm )
    for i in range(8):
        print('----%d-----' % i)
        print( torch.cat((R_r, T_r), dim=2)[i] )
        print( transform[i] )






