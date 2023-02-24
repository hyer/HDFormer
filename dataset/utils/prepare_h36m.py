# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import cdflib
from glob import glob
import numpy as np
import os
from os.path import join
from pathlib import Path

from dataset.lib import Human36mDataset, world_to_camera, project_to_2d, image_coordinates, wrap

home = str(Path.home())
path_root = join(home, 'Workspace/Dataset/h36m_position')
cdf_path = join(home, 'Workspace/Dataset/h36m')
path_3d = join(path_root, 'data_3d_h36m.npz')
path_3d_world = join(path_root, 'all_data_3d_world_h36m.npz')
path_2d = join(path_root, 'data_2d_h36m_gt.npz')
path_camera = join(path_root, 'data_camera.npz')
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

print('Converting original Human3.6M dataset from CDF files')
pose3d_world = {}

for subject in subjects:
    pose3d_world[subject] = {}
    file_list = glob(cdf_path + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')
    assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
    for f in file_list:
        action = os.path.splitext(os.path.basename(f))[0]
        if subject == 'S11' and action == 'Directions':
            continue  # Discard corrupted video
        # Use consistent naming convention
        canonical_name = action.replace('TakingPhoto', 'Photo') \
            .replace('WalkingDog', 'WalkDog')
        hf = cdflib.CDF(f)
        positions = hf['Pose'].reshape(-1, 32, 3)
        positions /= 1000  # Meters instead of millimeters
        pose3d_world[subject][canonical_name] = positions.astype('float32')
print('Saving all_data_3d_world_h36m ...')
np.savez_compressed(path_3d_world, positions_3d=pose3d_world)

#
print('Computing ground-truth 2D poses...')
dataset = Human36mDataset(path_3d_world)


def split_train_test(subs):
    output_2d_poses, output_3d_poses, output_cameras = [], [], []
    for subject in subs:
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)  # screen space [-1,1]
                output_2d_poses.append(pos_2d.astype('float32'))
                output_3d_poses.append(pos_3d.astype('float32'))
                output_cameras.append(cam)
    return output_3d_poses, output_2d_poses, output_cameras


d3, d2, c = split_train_test(['S1', 'S5', 'S6', 'S7', 'S8'])
data = {'pose_3d': d3, 'pose_2d': d2, 'camera': c}
print('==> saving training set...')
np.savez_compressed(join(path_root, 'training.npz'), data=data)

d3, d2, c = split_train_test(['S9', 'S11'])
data = {'pose_3d': d3, 'pose_2d': d2, 'camera': c}
print('==> saving testing set...')
np.savez_compressed(join(path_root, 'testing.npz'), data=data)
print('Done.')
