# SARNet: Semantic Augmented Registration of Large-Scale Urban Point Clouds

## Environments
The code mainly requires the following libraries and you can check `requirements.txt` for more environment requirements.
- PyTorch
- [nuscenes](https://github.com/nutonomy/nuscenes-devkit)

Please run the following commands to install `point_utils`
```
cd models/PointUtils
python setup.py install
```

## Datasets
After download the two datasets, we need to preprocess the datasets:
### [SemanticKITTI](http://www.semantic-kitti.org/)
We need to download both the points data and the semantic label,the initial dataset should be organized as:
```
SemanticKITTI
|-- velodyne/sequences
	|-- 00/velodyne
		|--000000.bin
		...
	|--01/velodyne
	...
|-- label/sequences
	|-- 00/labels
		|--000000.label
		...
	|--01/labels
	...
```
After using data_prepare_SemanticKITTI.py to preprocess the intial data, We write the coordinates and the semantic label of points into one file:
```
SemanticKITTI
|-- process
	|--train
		|--000000.npy
		...
	|--val
		|--0003178.npy
		...
	|--test
		|--0003632.npy
		...
```
### [NuScenes](https://www.nuscenes.org/)
Similar to SemanticKITTI, we need to download both the points data and the semantic label(Nusenes-lidarseg), 
and extract the lidarseg and v1.0-* folders to our nuScenes root directory, the initial dataset should be organized as:
```
NuScenes
|-- lidarseg
	|-- v1.0-{mini, test, trainval} <- Contains the .bin files; a .bin file contains the labels of the points in a point cloud
|-- samples	<-	Sensor data for keyframes.
	|-- LIDAR_TOP
|-- sweeps	<-	Sensor data for intermediate frames
	|-- LIDAR_TOP
|-- v1.0-{mini, test, trainval}	<- JSON tables that include all the meta data and annotations.

```
After using data_prepare_nuScenes.py to preprocess the intial data, We write the coordinates and the semantic label of points into one file:
```
NuScenes
|-- process
	|--train
		|--000000.npy
		...
	|--val
		|--0003178.npy
		...
	|--test
		|--0003632.npy
		...
```

## Training
We provide scripts to train both SemanticKITTI and NuScenes, you should specify your own dataset path, and you could use the default parameters or specify your own parameters like in the train_nuscenes.sh:
```
	sh script/train_kitti.sh 	->	train SemanticKITTI
	sh script/train_nuscenes.sh 	->	train NuScenes
```
	
## Testing
```
	sh script/test_kitti.sh 	->	test SemanticKITTI
	sh script/test_nuscenes.sh 	->	test NuScenes
```

## Acknowledgments
We want to thank all the following open-source projects for the help of the implementation:
- [RPMNet](https://github.com/yewzijian/RPMNet)
- [PointNet++](https://github.com/sshaoshuai/Pointnet2.PyTorch)(unofficial implementation, for Furthest Points Sampling)
- [RandLA-Net](https://github.com/aRI0U/RandLA-Net-pytorch.git)(unofficial implementation, for Semantic Segmentation)
