import pdb
import numpy as np
import os
from os.path import join
from mmcv.utils import get_logger
from pathlib import Path

from dataset.lib import Human36mDataset, world_to_camera, normalize_screen_coordinates, deterministic_random

home = str(Path.home())
path_root = join(home, 'Workspace/Dataset/h36m_processed')
path_3d = join(path_root, 'data_3d_h36m.npz')
path_2d = join(path_root, 'data_2d_h36m_gt.npz')

logger = get_logger(name='prepare_h36m')
assert os.path.isfile(path_3d) and os.path.isfile(path_2d), "Dataset file is missing!"
dataset = Human36mDataset(path_3d)
logger.info('<== Data loaded!')
logger.info('==> Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

logger.info('==> Loading 2D detections...')
keypoints = np.load(path_2d, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[
            subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue
        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

pose2d_all = keypoints
pose3d_all = dataset._data
cameras_all = dataset.cameras()


def split_train_test(subjects):
    logger.info('==> Fetching {}...'.format(subjects))
    out_pose3d = []
    out_pose2d = []
    out_camera_params = []
    for subject in subjects:
        for action in pose2d_all[subject].keys():
            poses_2d = pose2d_all[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_pose2d.append(poses_2d[i])
            if subject in cameras_all:
                cams = cameras_all[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
            if 'positions_3d' in pose3d_all[subject][action]:
                poses_3d = pose3d_all[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_pose3d.append(poses_3d[i])
    logger.info('Fetching Done! <==')
    return out_pose3d, out_pose2d, out_camera_params


d3, d2, c = split_train_test(['S1', 'S5', 'S6', 'S7', 'S8'])
data = {'pose_3d': d3, 'pose_2d': d2, 'camera': c}
logger.info('==> saving training set...')
np.savez_compressed(join(path_root, 'training.npz'), data=data)

d3, d2, c = split_train_test(['S9', 'S11'])
data = {'pose_3d': d3, 'pose_2d': d2, 'camera': c}
logger.info('==> saving testing set...')
np.savez_compressed(join(path_root, 'testing.npz'), data=data)

# pdb.set_trace()
# logger.info('==> saving processed data...')
# np.savez_compressed(join(path_root, 'h36m_pose_2d_gt.npz'), pose_2d=keypoints)
# np.savez_compressed(join(path_root, 'h36m_pose_3d_gt.npz'), pose_3d=dataset._data)
# np.savez_compressed(join(path_root, 'h36m_cameras.npz'), cameras=dataset.cameras())

# logger.info('==> checking saved data...')
# keypoints_saved = np.load(join(path_root, 'h36m_pose_2d_gt.npz'), allow_pickle=True)['pose_2d'].item()
# pose3d_saved = np.load(join(path_root, 'h36m_pose_3d_gt.npz'), allow_pickle=True)['pose_3d'].item()
# cameras_saved = np.load(join(path_root, 'h36m_cameras.npz'), allow_pickle=True)['cameras'].item()
