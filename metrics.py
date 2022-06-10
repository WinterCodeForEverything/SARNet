import numpy as np
import torch
from typing import Dict

from common.math_torch import se3
from common.torch import to_numpy



def compute_metrics( data: Dict, pred_transforms ) -> Dict:
   
    with torch.no_grad():
        gt_transforms = data['transform_gt']

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        
        metrics = {
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag)
        }

    return metrics

def summarize_metrics(metrics, rot_thres, trans_thres ):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    success_list = np.zeros((len(metrics['err_t'])), dtype=np.int32 )
    success_list[ metrics['err_r_deg'] < rot_thres] = 1
    success_list[ metrics['err_t'] > trans_thres] = 0
    metrics['err_r_deg_right'] = metrics['err_r_deg'][success_list==1]
    metrics['err_t_right'] = metrics['err_t'][success_list==1]
    success_rate = np.sum(success_list, dtype=np.float32) / len(success_list)
    summarized['success_rate'] = success_rate
    for k in metrics:
        if k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_std'] = np.std(metrics[k])
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized 

def print_metrics( logger, summary_metrics: Dict, title: str = 'Metrics' ):
    logger.info( title + ':' )
    logger.info('=' * (len(title) + 1))

    logger.info('Rotation error {:.4f}(deg, mean)+/-{:.4f}(std)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_std']))
    logger.info('Translation error {:.4g}(mean)+/-{:.4g}(std)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_std']))
    logger.info('error rotation in success {:.4f}(deg, mean)+/-{:.4f}(std)'.format(
        summary_metrics['err_r_deg_right_mean'], summary_metrics['err_r_deg_right_std']) )
    logger.info('error translation in success {:.4g}(mean)+/-{:.4g}(std)'.format(
        summary_metrics['err_t_right_mean'], summary_metrics['err_t_right_std']) )
    logger.info('success_rate {:.4g}'.format(summary_metrics['success_rate']))
