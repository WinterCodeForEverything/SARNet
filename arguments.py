import argparse
import os.path as osp

def dataset_arguments():
    parser = argparse.ArgumentParser( add_help=False )
    parser.add_argument('--dataset', type=str, default='SemanticKitti', choices=['SemanticKitti', 'NuScenes'], help='Which dataset to choose, the choise is SemanticKitti or NuScenes')
    parser.add_argument( '--sample_point_num', default=16000, type=int, help="The sampled points' number in the voxel sampling" )
    parser.add_argument( '--nsample', default=1024, type=int, help="The sampled points' number in the detector" )
    parser.add_argument('--sample_voxel_size', default = 0.3, type=float, help="The sampled voxel in the voxel sampling" )
    parser.add_argument('--boundingbox_diagonal', default = 102, type=float, help="The average diagonal of the dataset's boundingbox" )
    
    #SemanticKITTI
    parser.add_argument('--kitty_root', type=str, metavar='PATH', default = "/.../SemanticKITTY/process", help="Directory of SemanticKitti after preprocessing" )
    parser.add_argument('--kitti_ignore_label', default=[1,4,5,6,7,8], type=list, help="the semantic category in SemanticKitti will be ignored, sg, the moving category" )  
    #NuScenes
    parser.add_argument('--nuscenes_root', type=str, metavar='PATH', 
        default = '/.../nuscenes/process', help="Directory of Nuscenes after preprocessing" ) 
    parser.add_argument('--nuscenes_ignore_label', default=[1,2,3,4,5,7], type=list,help="the semantic category in Nuscenes will be ignored, sg, the moving category" )  
    return parser

def model_arguments():
    parser = argparse.ArgumentParser( add_help=False )
    parser.add_argument( '--nb', default=20, type=int, help="the neighbor points number in the k-nearest-neighboring" )
    parser.add_argument( '--init_dims', default=3, type=int, help="the input dimension of the network" )
    parser.add_argument( '--emb_dims', default=512, type=int, help="the embedding dimension of the network" )
    parser.add_argument( '--attention_head_num', default=4, type=int, help="multi-head attention number" )
    parser.add_argument('--trans_loss_type', type=str, choices=['mse', 'mae'], default='mae',
                        help=' Transformation loss to be optimized')
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--name', type=str, help='Prefix to add to logging directory')
    parser.add_argument('--semantic_classes_num', default = 20, type=int, help="Semantic classes number which wil be used in both segmention and registration" )
    parser.add_argument('--iter_num', default = 2, type=int, help=" iteration number in the RNN network" )
    parser.add_argument("--local_rank", type=int, help="Used in the mutil-gpu training")
    parser.add_argument('--rot_mag', default = 45.0, type=float, help="Rotation limitation in the data processing" )
    parser.add_argument('--trans_mag', default = 5.0, type=float, help="Translation limitation in the data processing" )
    parser.add_argument('--partial_p_keep', default = [0.7,0.7], type=list, help="The ratio of keeping part in the croping processing" )
    return parser


def train_arguments():
    parser = argparse.ArgumentParser( parents=[dataset_arguments(), model_arguments()] )
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate during training')
    parser.add_argument('--scheduler_step_size', default=10, type=int, help='The scheduler step size, the learning rate will decrease every * epoches' )
    parser.add_argument('--scheduler_gamma', default=0.5, type=float, help='The reduced speed of the learning rate' )
    parser.add_argument('--augment', default = 0.5, type=float,
            help="The probability the data will be extra processed, eg, add another random rotation and translation to moving categories" )
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data_loader loader')
    parser.add_argument( '--epoch_num',  default=30, type=int,  metavar='N', help="The epoch number in the training" )
    parser.add_argument('--train_batch_size', default=3, type=int, metavar='N',
                    help='Training mini-batch size(default 8)')
    parser.add_argument( '--val_batch_size', default=12, type=int, metavar='N',
                    help='The mini-batch size during validation or testing')
    parser.add_argument( '--noise_type',  default='jitter', type=str, help="Whether to add extra noise to the data" )
    parser.add_argument( '--save_checkpoints_path', type=str, 
                    default=osp.join( '/.../checkpoints', 'ckpt' ), help="Directory to save checkpoints" )
    parser.add_argument( '--load_checkpoints_path', type=str, 
                    default=osp.join('/.../checkpoints', 'ckpt-best.pth'), help="Directory to load checkpoints"  )
    parser.add_argument( '--validate_every', default=-1, type=int,
                    help="The step size for validation, negetive number means validating every * epoches" )
    parser.add_argument( '--summary_every', default=200, type=int, help="The step size for summary, negetive number means summary every * epoches"  )
    parser.add_argument('--RRE_thresholds', default = 2.0, type=float,
                    help="The relative rotation errer thresholds to judge whether it's successful registration" )
    parser.add_argument('--RTE_thresholds', default = 0.5, type=float,
                    help="The relative translation errer thresholds to judge whether it's successful registration" )
    parser.add_argument('--logdir', default='/.../logs', type=str,
                        help='Directory to store logs, summaries, checkpoints.')
    return parser

def test_arguments():
    parser = argparse.ArgumentParser( parents=[dataset_arguments(), model_arguments()] )
    parser.add_argument('--test_batch_size', default=16, type=int, metavar='N',
                        help='The test mini-batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for data_loader loader.')
    parser.add_argument('--eval_save_path', type=str, default='./eval_results',
                        help='Output data_loader to save evaluation results')
    parser.add_argument('--transform_file', type=str,
                        help='If provided, will use transforms from this provided pickle file')
    parser.add_argument('--RRE_thresholds', default = 2.0, type=float,
                        help="The relative rotation errer thresholds to judge whether it's successful registration" )
    parser.add_argument('--RTE_thresholds', default = 0.5, type=float,
                        help="The relative translation errer thresholds to judge whether it's successful registration" )
    parser.add_argument('--checkpoints_path', default='/.../checkpoints', type=str,
                        help='Directory to load checkpoints.')
    parser.add_argument('--augment', default = 0.5, type=float,
                        help="the probability the data will be extra processed, eg, add another random rotation and translation to moving categories" )
    parser.add_argument( '--noise_type',  default='clean', type=str,
                        help="Whether to add extra noise to the data" )
    #parser.add_argument( '--transform_file',  default=None, type=str, help="" )
    return parser


if __name__ == '__main__':
    parser = train_arguments()
    _args = parser.parse_args()
