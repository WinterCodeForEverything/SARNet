import os
import numpy as np
from os.path import join, exists
#import h5py
from collections import defaultdict
from nuscenes import NuScenes

data_root = 'your nuscense path'
process_data_path = join( data_root, 'process')
if not exists(process_data_path):
    os.makedirs(process_data_path)
train_data_path = join( process_data_path, 'train')
if not exists(train_data_path):
    os.makedirs(train_data_path)
val_data_path = join( process_data_path, 'val')
if not exists(val_data_path):
    os.makedirs(val_data_path)
test_data_path = join( process_data_path, 'test')
if not exists(test_data_path):
    os.makedirs(test_data_path)


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)
    scene_sample_data = defaultdict(list)
    for lidarseg in nusc.lidarseg:
        sample_data = nusc.get('sample_data', lidarseg['sample_data_token'])
        points_fn = join(data_root, sample_data['filename'])
        sample = nusc.get('sample', sample_data['sample_token'])
        scene =  nusc.get('scene', sample['scene_token'])
        scene_sample_data[scene['name']].append( [sample_data['filename'], 
            lidarseg['filename'], sample_data['token']] )
    for k in scene_sample_data:
        sample_data_idx = 0
        for s in scene_sample_data[k]:
            points_fn, seg_fn, data_token = s[0], s[1], s[2]
            points = np.fromfile(join(data_root, points_fn), dtype=np.float32, count=-1).reshape([-1,5])
            xyz = points[:, :3]
            label = np.fromfile(join(data_root, seg_fn), dtype=np.uint8, count=-1).reshape([-1,1]).astype(np.float32)
            data = np.concatenate((xyz, label), axis=1)     
            if sample_data_idx < 30:
                save_full_path = join( train_data_path, data_token + '.npy')
                np.save( save_full_path, data )
            elif sample_data_idx < 34:
                save_full_path = join( val_data_path, data_token + '.npy')
                np.save( save_full_path, data )
            else:
                save_full_path = join( test_data_path, data_token + '.npy')
                np.save( save_full_path, data )  
            print( data_token )                      
            sample_data_idx += 1
