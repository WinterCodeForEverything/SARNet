import os
import open3d as o3d
import numpy as np
from os.path import join, exists
from torch.utils.data import Dataset

import yaml

import common.math.se3 as se3 
import common.math.random as rdm 

label_yaml_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'semantic-nuscenes.yaml' )
DATA = yaml.safe_load(open(label_yaml_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 20), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class NuScenesDataSet(Dataset):
    def __init__(self, root, transform, split = 'train', ignore_label=None, augment = 0.5 ):
        super(NuScenesDataSet, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.dataset = self.make_dataset()
        self.ignore_label = ignore_label
        self.augment = augment

    def make_dataset(self):
        data_path = join( self.root, self.split )
        dataset = os.listdir(data_path)
        return dataset

    def __getitem__(self, index):
        fn = self.dataset[index]
        data = np.load(join( self.root, self.split, fn ))
        points = data[:,:3]
        label = data[:, 3]
        sample = { 'points' : points.astype('float32'), 'seg' : remap_lut[label.astype('int32')],
             'idx': np.array(index, dtype=np.int32) }
        sample = self.transform( sample )
        points_src, points_ref = sample['points_src'], sample['points_ref']
        labels_src, labels_ref = sample['seg_src'], sample['seg_ref']
        intersect_elm = np.intersect1d( labels_src, labels_ref )
        if self.ignore_label != None:
            intersect_elm = np.setdiff1d( intersect_elm, self.ignore_label )
            if np.random.rand() < self.augment:
                for il in self.ignore_label:
                    rand_T = np.zeros((4,4), dtype=np.float32)
                    rand_T[3,3] = 1.0
                    rand_rotm = rdm.generate_rand_rotm( 3.0, 3.0, 3.0 ) 
                    rand_T[:3,:3] = rand_rotm
                    rand_trans = rdm.generate_rand_trans( 10.0, 1.0, 0.1 )
                    rand_T[:3,3] = rand_trans
                    points_src[labels_src == il] = se3.transform( rand_T,
                         points_src[labels_src == il] )

        intersect_src = np.isin( labels_src, intersect_elm ).astype(int)
        intersect_ref = np.isin( labels_ref, intersect_elm ).astype(int)     
        sample['intersect_src'] = intersect_src
        sample['intersect_ref'] = intersect_ref

        sample['points_src'] = points_src.astype('float32')
        sample['points_ref'] = points_ref.astype('float32')

        sample.pop( "seg" )
        sample.pop( "points_raw" )
        return sample
    
    def __len__(self):
        return len(self.dataset)

