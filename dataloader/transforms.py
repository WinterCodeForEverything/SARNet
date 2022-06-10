import math
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data

import open3d as o3d

from common.math.random import uniform_2_sphere
import common.math.se3 as se3
import common.math.so3 as so3

def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 5, voxel_size: float = 0.3, 
                   num: int = 1024, diagonal: float = 1.0, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        voxel_size: voxel cell size to do voxel sampling
        num: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [VoxelResampler(voxel_size, num),
                            SplitSourceRef(),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            ShufflePoints()]

        test_transforms = [SetDeterministic(),
                           FixedVoxelResampler(voxel_size, num),
                           SplitSourceRef(),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [SplitSourceRef(),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            VoxelResampler(voxel_size, num),
                            RandomJitter(diagonal),
                            ShufflePoints()]

        test_transforms = [SetDeterministic(),
                           SplitSourceRef(),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           VoxelResampler(voxel_size, num),
                           RandomJitter(diagonal),
                           ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [SplitSourceRef(),
                            RandomCrop(partial_p_keep),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            VoxelResampler(voxel_size, num),
                            RandomJitter(diagonal),
                            ShufflePoints()]

        test_transforms = [SetDeterministic(),
                           SplitSourceRef(),
                           RandomCrop(partial_p_keep),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           VoxelResampler(voxel_size, num),
                           RandomJitter(diagonal),
                           ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms



class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
            if 'seg' in sample:
                sample['seg_src'] = sample['seg'].detach()
                sample['seg_ref'] = sample['seg'].detach()

        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()
            if 'seg' in sample:
                sample['seg_src'] = sample['seg'].copy()
                sample['seg_ref'] = sample['seg'].copy()

        return sample


class VoxelResampler:
    def __init__(self, voxel_size: float, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.voxel_size = voxel_size
        self.num = num
        

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            if 'seg' in sample:
                sample['points'], sample['seg'] = self._resample(sample['points'], sample['seg'], self.voxel_size, self.num)
            else:
                sample['points'], _ = self._resample(sample['points'], None, self.voxel_size, self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            if 'seg' in sample:
                sample['points_src'], sample['seg_src'] = self._resample(sample['points_src'], sample['seg_src'], 
                        self.voxel_size, src_size )
                sample['points_ref'], sample['seg_ref'] = self._resample(sample['points_ref'], sample['seg_ref'], 
                        self.voxel_size, ref_size )
            else:
                sample['points_src'], _ = self._resample(sample['points_src'], None, self.voxel_size, src_size )
                sample['points_ref'], _ = self._resample(sample['points_ref'], None, self.voxel_size, ref_size )                

        return sample

    @staticmethod
    def _resample( points, seg, voxel_size, k ):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if voxel_size is not None:
            pcd_ds_and_idx = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False )
            points = np.asarray( pcd_ds_and_idx[0].points)
            idx = np.max( pcd_ds_and_idx[1], axis=1 )
            if seg is not None:
                seg = seg[idx]           

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            if seg is not None:
                return points[rand_idxs, :], seg[rand_idxs]
            return points[rand_idxs, :], None
        elif points.shape[0] == k:
            return points, seg
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            if seg is not None:
                return points[rand_idxs, :], seg[rand_idxs]
            return points[rand_idxs, :], None


class FixedVoxelResampler(VoxelResampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, seg, voxel_size, k):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if voxel_size is not None:
            pcd_ds_and_idx = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False )
            points = np.asarray( pcd_ds_and_idx[0].points)
            idx = np.max( pcd_ds_and_idx[1], axis=1 )
            if seg is not None:
                seg = seg[idx]     
         
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        points = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        if seg is not None:
            seg = np.concatenate((np.tile(seg, multiple), seg[:remainder]), axis=0)
        return points, seg


class RandomJitter:
    """ generate perturbations """
    def __init__(self, diagonal= 1):   #scale=0.01, clip=0.05
        self.scale = diagonal * 0.01
        self.clip = diagonal * 0.05

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class RandomCrop:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample


class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomRotatorZ(RandomTransformSE3):
    """Applies a random z-rotation to the source point cloud"""

    def __init__(self):
        super().__init__(rot_mag=360)

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        rand_rot_deg = np.random.random() * self._rot_mag
        rand_rot = Rotation.from_euler('z', rand_rot_deg, degrees=True).as_dcm()
        rand_SE3 = np.pad(rand_rot, ((0, 0), (0, 1)), mode='constant').astype(np.float32)

        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            per_arr = np.random.permutation( sample['points'].shape[0] )
            sample['points'] = sample['points'][per_arr, :]
            if 'seg' in sample:
                sample['seg'] = sample['seg'][per_arr]
        else:
            src_arr = np.random.permutation( sample['points_src'].shape[0] )
            sample['points_src'] = sample['points_src'][src_arr, :]
            ref_arr = np.random.permutation( sample['points_ref'].shape[0] )
            sample['points_ref'] = sample['points_ref'][ref_arr, :]
            if 'seg' in sample:
                sample['seg_src'] = sample['seg_src'][src_arr]
                sample['seg_ref'] = sample['seg_ref'][ref_arr]
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class Dict2DcpList:
    """Converts dictionary of tensors into a list of tensors compatible with Deep Closest Point"""
    def __call__(self, sample):

        target = sample['points_src'][:, :3].transpose().copy()
        src = sample['points_ref'][:, :3].transpose().copy()

        rotation_ab = sample['transform_gt'][:3, :3].transpose().copy()
        translation_ab = -rotation_ab @ sample['transform_gt'][:3, 3].copy()

        rotation_ba = sample['transform_gt'][:3, :3].copy()
        translation_ba = sample['transform_gt'][:3, 3].copy()

        euler_ab = Rotation.from_dcm(rotation_ab).as_euler('zyx').copy()
        euler_ba = Rotation.from_dcm(rotation_ba).as_euler('xyz').copy()

        return src, target, \
               rotation_ab, translation_ab, rotation_ba, translation_ba, \
               euler_ab, euler_ba


class Dict2PointnetLKList:
    """Converts dictionary of tensors into a list of tensors compatible with PointNet LK"""
    def __call__(self, sample):

        if 'points' in sample:
            # Train Classifier (pretraining)
            return sample['points'][:, :3], sample['label']
        else:
            # Train PointNetLK
            transform_gt_4x4 = np.concatenate([sample['transform_gt'],
                                               np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
            return sample['points_src'][:, :3], sample['points_ref'][:, :3], transform_gt_4x4
