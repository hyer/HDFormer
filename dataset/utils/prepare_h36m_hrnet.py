from os.path import join
from pathlib import Path

import numpy as np
import os
import h5py
# import logging
import torch.utils.data

from dataset.lib import cameras
from dataset.lib import get_skeleton
from base.utilities import get_logger

# h36m has 32 joints, 17 of which are used.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

data_path = 'Dataset/h36m_processed/data/'
subject_ids = [1, 5, 6, 7, 8, 9, 11]
logger = get_logger()

assert os.path.isfile(data_path + "train.h5") and os.path.isfile(
    data_path + "test.h5"), "Dataset file is missing!"
with h5py.File(data_path + "train.h5", "r") as f:
    train_set_3d = {}
    for k in f["data_3d"]:
        d = f["data_3d"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        train_set_3d[key] = d[:]
    train_set_2d_gt = {}
    for k in f["data_2d_gt"]:
        d = f["data_2d_gt"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        train_set_2d_gt[key] = d[:]

with h5py.File(data_path + "test.h5", "r") as f:
    test_set_3d = {}
    for k in f["data_3d"]:
        d = f["data_3d"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        test_set_3d[key] = d[:]
    test_set_2d_gt = {}
    for k in f["data_2d_gt"]:
        d = f["data_2d_gt"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        test_set_2d_gt[key] = d[:]

logger.info(
    "{} 3d train files, {} 3d test files are loaded.".format(len(train_set_3d), len(test_set_3d)))
logger.info("{} 2d GT train files, {} 2d GT test files are loaded.".format(len(train_set_2d_gt),
                                                                           len(test_set_2d_gt)))

data_set_3d = {**train_set_3d, **test_set_3d}
data_set_2d_gt = {**train_set_2d_gt, **test_set_2d_gt}
# data_set_3d = train_set_3d
# data_set_2d_gt = train_set_2d_gt

f_cpn = np.load(data_path + "data_cpn.npz", allow_pickle=True)
data_2d_cpn = f_cpn["positions_2d"].item()

a = np.load(data_path + 'twoDPose_HRN_test.npy', allow_pickle=True).item()
b = np.load(data_path + 'twoDPose_HRN_train.npy', allow_pickle=True).item()
# ab = {**a, **b}
# data_2d_hr = {}
# for k in ab:
#     d = ab[k]
#     key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
#     data_2d_hr[key] = d[:]
data_2d_hr = {**a, **b}
dataset_2d_hr = {}
for k in data_2d_hr:
    k_hr = []
    k_hr.append(k[0])
    k_hr.append(k[1])
    k_hr.append(k[2][:-3])
    k_hr = tuple(k_hr)
    dataset_2d_hr[k_hr] = data_2d_hr[k]

# #############################################################################
data_d2_gt = {}
data_d2_cpn = {}
data_d2_hr = {}
data_d3 = {}
indices = []

# cut videos into short clips of fixed length
logger.info("Loading sequence...")
dims_17 = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
dim_2d = np.sort(np.hstack([dims_17 * 2 + i for i in range(2)]))
dim_3d = np.sort(np.hstack([dims_17 * 3 + i for i in range(3)]))

for idx, k in enumerate(sorted(data_set_3d)):
    if k[0] == 11 and k[2].split(".")[0] == "Directions":
        # one video is missing
        # drop all four videos instead of only one camera's view
        data_d3[k] = None
        continue
    # from IPython import embed; embed()
    assert k in data_set_2d_gt, k
    assert data_set_3d[k].shape[0] == data_set_2d_gt[k].shape[0]

    cam_name = k[2].split(".")[1]
    cam_id = cameras.cam_name_to_id[cam_name]
    d2_cpn = data_2d_cpn["S{}".format(k[0])][k[2].split(".")[0]][cam_id - 1][:data_set_3d[k].shape[0], :]
    d2_cpn = d2_cpn.reshape([d2_cpn.shape[0], 17 * 2])
    data_d2_cpn[k] = d2_cpn.reshape([d2_cpn.shape[0], 17, 2])  # [T,V,C]

    d2_gt = data_set_2d_gt[k][:, dim_2d]
    d2_gt = d2_gt.reshape([d2_gt.shape[0], 17, 2])
    d2_gt = d2_gt.reshape([d2_gt.shape[0], 17 * 2])
    data_d2_gt[k] = d2_gt.reshape([d2_gt.shape[0], 17, 2])  # [T,V,C]

    d2_hr = dataset_2d_hr[k][:, dim_2d]
    d2_hr = d2_hr.reshape([d2_hr.shape[0], 17, 2])
    assert d2_hr.shape[0] == d2_gt.shape[0], (d2_hr.shape, d2_gt.shape)
    data_d2_hr[k] = d2_hr

cpn_err = np.concatenate(list(data_d2_cpn.values()), axis=0) - np.concatenate(list(data_d2_gt.values()),
                                                                              axis=0)
cpn_err = np.linalg.norm(cpn_err, axis=-1).mean()
logger.info('==> 2D data error: %.2f' % cpn_err)

hr_err = np.concatenate(list(data_d2_hr.values()), axis=0) - np.concatenate(list(data_d2_gt.values()),
                                                                            axis=0)
hr_err = np.linalg.norm(hr_err, axis=-1).mean()
logger.info('==> HR 2D data error: %.2f' % hr_err)
np.savez_compressed(data_path + 'data_hrnet.npz', data=data_d2_hr)
