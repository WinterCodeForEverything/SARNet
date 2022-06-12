# SARNet: Semantic Augmented Registration of Large-Scale Urban Point Clouds

## Environments
The code mainly requires the following libraries and you can check `requirements.txt` for more environment requirements.
- PyTorch
- nuscenes( )

Please run the following commands to install `point_utils`
```
cd models/PointUtils
python setup.py install
```

## Datasets
After download the two datasets, we need to preprocess the datasets:
### SemanticKITTI
download url: 
the initial dataset is:
After preprocessing, the dataset is:
### NuScenes
download url:
the initial dataset is:
After preprocessing, the dataset is:

## Training

## Testing

## Acknowledgments
We want to thank all the following open-source projects for the help of the implementation:
- [RPMNet] (https://github.com/yewzijian/RPMNet)
- [PointNet++](https://github.com/sshaoshuai/Pointnet2.PyTorch)(unofficial implementation, for Furthest Points Sampling)
- [RandLA-Net] (https://github.com/aRI0U/RandLA-Net-pytorch.git) (unofficial implementation, for Semantic segmentation)
