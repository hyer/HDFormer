from os.path import join
from pathlib import Path
import numpy as np
import os
import h5py
from base.utilities import get_logger
import matplotlib.pyplot as plt
from scipy.stats import norm

from dataset.lib import cameras
from dataset.Human36M import H36M_NAMES

# Config
logger = get_logger()
home = str(Path.home())
base_path = join(home, 'Workspace/Dataset/h36m_processed/data')
gt_test_path = join(base_path, 'test.h5')
pred_path = join(base_path, 'data_hrnet.npz')
cameras_path = join(base_path, 'cameras.h5')
n_joints = 17

# Loading data
subject_ids = [1, 5, 6, 7, 8, 9, 11]
rcams = cameras.load_cameras(cameras_path, subject_ids)
cam_name_to_id = {
    "54138969": 1,
    "55011271": 2,
    "58860488": 3,
    "60457274": 4,
}
with h5py.File(gt_test_path, "r") as f:
    data_3d = {}
    for k in f["data_3d"]:
        d = f["data_3d"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        data_3d[key] = d[:]
    data_2d_gt = {}
    for k in f["data_2d_gt"]:
        d = f["data_2d_gt"][k]
        key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
        data_2d_gt[key] = d[:]

pred = np.load(pred_path, allow_pickle=True)["data"].item()
logger.info("Loading sequence...")
dims_17 = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
dim_3d = np.sort(np.hstack([dims_17 * 3 + i for i in range(3)]))
dim_2d = np.sort(np.hstack([dims_17 * 2 + i for i in range(2)]))

structured_3d, structured_2d_gt, structured_2d_pred, centered_2d_gt = {}, {}, {}, {}

for idx, k in enumerate(sorted(data_3d)):
    if k[0] == 11 and k[2].split(".")[0] == "Directions":
        # one video is missing
        # drop all four videos instead of only one camera's view
        structured_3d[k] = None
        continue
    # from IPython import embed; embed()
    assert k in data_2d_gt, k
    assert data_3d[k].shape[0] == data_2d_gt[k].shape[0]
    cam_name = k[2].split(".")[1]
    cam_id = cameras.cam_name_to_id[cam_name]
    subject_idx = k[0]
    camera_parameters = rcams[(subject_idx, cam_id)]
    f, c = camera_parameters[2], camera_parameters[3]
    f, c = np.expand_dims(f.transpose(1, 0), 0), np.expand_dims(c.transpose(1, 0), 0)

    d2_gt = data_2d_gt[k][:, dim_2d]
    structured_2d_gt[k] = d2_gt.reshape([d2_gt.shape[0], n_joints, 2])
    structured_2d_pred[k] = pred[k]  # [T,V,C]
    centered_2d_gt[k] = d2_gt.reshape([d2_gt.shape[0], n_joints, 2]).copy()
    centered_2d_gt[k] = (centered_2d_gt[k] - 0.5 - c) / f
logger.info("Loading %d sequences done!" % len(structured_2d_gt))

# Analysis
d2_err = np.concatenate(list(structured_2d_pred.values()), axis=0) \
         - np.concatenate(list(structured_2d_gt.values()), axis=0)
err = np.linalg.norm(d2_err, axis=-1).mean()
logger.info('==> 2D data error: %.2f' % err)

error = d2_err
num_joint = n_joints
name_joint = ['Hip', \
              'RightHip', 'RightKnee', 'RightFoot', \
              'LeftHip', 'LeftKnee', 'LeftFoot', \
              'Spine', 'Neck', 'Head', 'Site', \
              'LeftShoulder', 'LeftElbow', 'LeftHand', \
              'RightShoulder', 'RightElbow', 'RightHand']

# create save directory
save_dir = 'vis_res/stat_mix_hrnet'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ----------------------------------------------------------------------
# parameter estimation using EM
mean = np.zeros((num_joint, 2))
std = np.zeros((num_joint, 2))
weight = np.zeros((num_joint))
for j in range(num_joint):
    e = error[:, j, :]

    # non-robust estimation of std (we assume zero mean)
    m1 = np.mean(e, axis=0)
    s1 = np.sqrt(np.mean((e - m1) ** 2.0, axis=0))

    # robust estimation of std (we assume zero mean)
    q75 = np.percentile(e, 75, axis=0)
    q25 = np.percentile(e, 25, axis=0)
    m2 = np.percentile(e, 50, axis=0)
    s2 = 0.7413 * (q75 - q25)

    # initial estimate
    m = m2
    s = s2
    w = 0.5
    print(m, s, w)

    # NLL log
    NLL = np.zeros((10))

    for k in range(10):
        # E-step: compute responsibility
        p1 = w * np.exp(-np.sum(((e - m) ** 2.0) / (2 * s ** 2.0), axis=1, keepdims=True)) / (2 * np.pi * s[0] * s[1])
        p2 = (1.0 - w) * 1 / 10000.0
        r1 = p1 / (p1 + p2)
        r2 = p2 / (p1 + p2)

        # M-step: re-estimate the parameters
        m = np.sum(r1 * e, axis=0) / np.sum(r1)
        s = np.sqrt(np.sum(r1 * (e - m) ** 2.0, axis=0) / np.sum(r1))
        w = np.mean(r1)

        # compute NLL
        nll = -np.mean(np.log(p1 + p2))
        NLL[k] = nll

        #
        print(m, s, w, nll)

    #
    mean[j] = m
    std[j] = s
    weight[j] = w

    #
    tick_x_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tick_x_lab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.clf()
    plt.plot(np.linspace(1, 10, 10), NLL, '-s', lw=1.5, markeredgecolor='k', markerfacecolor='r')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.title('%s' % (name_joint[j]))
    plt.xticks(tick_x_val, tick_x_lab)
    plt.grid(True)
    plt.savefig('%s/em_%s.pdf' % (save_dir, name_joint[j]))

    #
    for i in range(2):
        pdf = lambda x: np.exp(-((x - m[i]) ** 2.0) / (2 * s[i] ** 2.0)) * w / np.sqrt(2 * np.pi * s[i] ** 2.0) + (
                1 - w) / 100.0

        tick_x_val = [-20, -10, 0, 10, 20]
        tick_x_lab = ['-20', '-10', '0', '10', '20']

        plt.clf()
        plt.hist(e[:, i], bins=100, histtype='stepfilled', range=[-20.0, 20.0], density=True)

        # s_size = 10000000
        # noise = np.zeros([2, s_size, 1])
        # noise[1, ...] = np.random.uniform(-50, 50, size=[s_size, 1])
        # noise[0, ...] = np.random.normal(loc=m[i], scale=s[i], size=(s_size, 1))
        # random_idx = np.random.choice(np.arange(2), size=(s_size,), p=np.array([w, 1.0 - w]))
        # noise = noise[random_idx, np.arange(s_size), :]
        # plt.hist(noise, bins=100, histtype='stepfilled', range=[-20.0, 20.0], density=True)

        x_sample = np.linspace(-20, 20, 1000)
        plt.plot(x_sample, norm(m1[i], s1[i]).pdf(x_sample), '--g', lw=1.5,
                 label='Gaussian Fit')
        # plt.plot(x_sample, norm(m2[i], s2[i]).pdf(x_sample), ':k', lw=1.5,
        #         label='robust gaussian fit')
        plt.plot(x_sample, pdf(x_sample), '-r', lw=1.5,
                 label='Mixture Fit')
        plt.legend()

        plt.xlabel('Error (in pixels)')
        plt.ylabel('Probability')
        if i == 0:
            plt.title('%s (x)' % (name_joint[j]))
        elif i == 1:
            plt.title('%s (y)' % (name_joint[j]))
        plt.xticks(tick_x_val, tick_x_lab)
        plt.grid(True)
        if i == 0:
            plt.savefig('%s/error_%s_x.png' % (save_dir, name_joint[j]))
            plt.savefig('%s/error_%s_x.pdf' % (save_dir, name_joint[j]))
        elif i == 1:
            plt.savefig('%s/error_%s_y.png' % (save_dir, name_joint[j]))
            plt.savefig('%s/error_%s_y.pdf' % (save_dir, name_joint[j]))

# save results
print(mean)
print(std)
print(weight)
err_stat = {'mean': mean, 'std': std, 'weight': weight}
center_data_stat = {}
c_data = np.concatenate(list(centered_2d_gt.values()), axis=0)
center_data_stat['mean'] = np.mean(c_data, axis=0)
center_data_stat['std'] = np.std(c_data, axis=0)
np.savez_compressed(join(save_dir, 'hrnet.npz'), data=structured_2d_pred, center_data_stat=center_data_stat,
                    error=error, err_stat=err_stat)
