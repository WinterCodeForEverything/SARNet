import os
import numpy as np
from os.path import join, exists

data_root = 'your SemanticKitty path'
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
   for i in range(11):
        points_file_path = os.path.join(data_root, 'velodyne/sequences', f'{i:02d}', 'velodyne' )
        labels_file_path = os.path.join(data_root, 'label/sequences', f'{i:02d}', 'labels' )
        points_file_names = sorted(os.listdir(points_file_path))
        file_num = len(points_file_names)
        train_num, val_num = file_num*7//10, file_num*8//10
        file_idx = 0
        for pfn in points_file_names:
            bpfn = os.path.splitext(pfn)[0]
            pffn = os.path.join(points_file_path, pfn)
            lfn = bpfn + '.label'
            lffn = os.path.join(labels_file_path, lfn)
            points = np.fromfile( pffn, dtype=np.float32, count=-1).reshape([-1,4] )[:, :3]
            label = np.fromfile( lffn, dtype=np.uint32 ).reshape( (-1,1) )
            label = label & 0xFFFF  # only semantic label
            label = label.astype(np.float32)
            data = np.concatenate((points, label), axis=1)
            if file_idx < train_num:
                save_full_path = join( train_data_path, f'{i:02d}' + bpfn + '.npy')
                np.save( save_full_path, data )
            elif file_idx < val_num:
                save_full_path = join( val_data_path, f'{i:02d}' + bpfn + '.npy')
                np.save( save_full_path, data ) 
            else:
                save_full_path = join( test_data_path, f'{i:02d}' + bpfn + '.npy')
                np.save( save_full_path, data ) 
            print( f'{i:02d}' + bpfn + '.npy' )
            file_idx += 1