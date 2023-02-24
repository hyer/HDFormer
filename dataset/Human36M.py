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


class H36M_Dataset(torch.utils.data.Dataset):
    """
    dataset return instance: [C,T,V]
    dataloader return instance: [B,C,T,V]
    """

    def __init__(self, config, mode, logger):
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = logger
        assert mode in ["train", "test"], "Invalid mode: {}".format(mode)
        # home = str(Path.home())
        config.data_path = str(config.data_path) + '/'
        config.cameras_path = join(config.data_path, 'cameras.h5')
        self.config = config
        self.mode = mode
        self.skeleton = get_skeleton()

        subject_ids = [1, 5, 6, 7, 8, 9, 11]
        rcams = cameras.load_cameras(config.cameras_path, subject_ids)
        self.rcams = rcams
        self.cam_name_to_id = {
            "54138969": 1,
            "55011271": 2,
            "58860488": 3,
            "60457274": 4,
        }
        self.valid_length = {}

        assert os.path.isfile(config.data_path + "train.h5") and os.path.isfile(
            config.data_path + "test.h5"), "Dataset file is missing!"
        path = config.data_path + "train.h5" if mode == 'train' else config.data_path + "test.h5"
        with h5py.File(path, "r") as f:
            data_3d = {}
            for k in f["data_3d"]:
                d = f["data_3d"][k]
                key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                data_3d[key] = d[:]
            data_2d = {}
            for k in f["data_2d_gt"]:
                d = f["data_2d_gt"][k]
                key = d.attrs["subject"], d.attrs["action"], d.attrs["filename"]
                data_2d[key] = d[:]
        assert len(data_3d) == len(data_2d), str(len(data_3d)) + 'mismatch' + str(len(data_2d))
        self.logger.info("%d files loaded!" % len(data_3d))

        if config.test_d2_type != 'data_2d_gt':
            self.aug = True
            f = np.load(config.data_path + "%s.npz" % config.test_d2_type, allow_pickle=True)
            self.center_data_stat = f['center_data_stat'].item()
            if mode == 'test':
                data_2d = f['data'].item()
            else:
                self.noise = f['error']
                self.noise_stat = f['err_stat'].item()
        else:
            self.aug = False
            f = np.load(config.data_path + "tcn_hrnet.npz", allow_pickle=True)
            self.center_data_stat = f['center_data_stat'].item()

        self.n_frames = config.n_frames
        self.n_joints = config.n_joints
        self.window_slide = config.window_slide

        self.left_right_symmetry_2d = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        # self.left_right_symmetry_3d = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12])

        self.data_2d = {}
        self.data_3d = {}
        self.indices = []

        # cut videos into short clips of fixed length
        self.logger.info("Loading sequence...")
        dims_17 = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        assert self.n_joints == 17, self.n_joints
        dim_2d = np.sort(np.hstack([dims_17 * 2 + i for i in range(2)]))
        dim_3d = np.sort(np.hstack([dims_17 * 3 + i for i in range(3)]))

        for idx, k in enumerate(sorted(data_3d)):
            if k[0] == 11 and k[2].split(".")[0] == "Directions":
                # one video is missing
                # drop all four videos instead of only one camera's view
                self.data_3d[k] = None
                continue
            d3 = data_3d[k][:, dim_3d]
            d3 = d3.reshape([d3.shape[0], self.n_joints, 3])
            # align root to origin
            d3 = d3 - d3[:, :1, :]
            # remove zero root joint
            # d3 = d3[:, 1:, :]
            self.data_3d[k] = d3  # [T,V,C]

            if mode == 'test' and config.test_d2_type != 'data_2d_gt':
                d2 = data_2d[k]
            else:  # train
                d2 = data_2d[k][:, dim_2d].reshape([d3.shape[0], self.n_joints, 2])
            self.data_2d[k] = d2  # [T,V,C]

            N = data_3d[k].shape[0]
            n = 0
            while n + self.n_frames <= N:
                self.indices.append((idx,) + k + (n, self.n_frames))
                n += self.window_slide
            valid_length = n - self.window_slide + self.n_frames
            self.valid_length[k] = valid_length

        self.n_data = len(self.indices)
        self.logger.info("{} data loaded for {} dataset".format(self.n_data, mode))
        self.logger.info('==> Using %s testing' % config.test_d2_type)

        # computing statistics for data normalization
        if hasattr(config, 'states'):
            assert mode == "test", mode
            stats_data = config.states
            self.logger.info("Loading stats...")
            self.mean_3d, self.std_3d = stats_data
        else:
            assert mode == "train", mode
            data = np.concatenate(list(self.data_3d.values()), axis=0)  # take care joint 0
            self.mean_3d = np.mean(data,
                                   axis=0 if hasattr(config, 'PJN') and config.PJN else (0, 1))
            self.std_3d = np.std(data,
                                 axis=0 if hasattr(config, 'PJN') and config.PJN else (0, 1))
            self.logger.info("mean 3d: {}".format(self.mean_3d))
            self.logger.info("std 3d: {}".format(self.std_3d))
            stats_data = self.mean_3d, self.std_3d
            config.states = stats_data
            self.logger.info("Saving stats...")

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        assert 0 <= item < self.n_data, "Index {} out of range [{}, {})".format(item, 0, self.n_data)

        index = self.indices[item]
        idx, k0, k1, k2, n, t = index
        camera_index = self.cam_name_to_id[k2.split('.')[1]]
        camera_parameters = self.rcams[(k0, camera_index)]
        f, c = camera_parameters[2], camera_parameters[3]
        f, c = np.expand_dims(f.transpose(1, 0), 0), np.expand_dims(c.transpose(1, 0), 0)
        data_3d = self.data_3d[k0, k1, k2][n: n + t].copy()
        data_2d = self.data_2d[k0, k1, k2][n: n + t].copy()
        # AUG
        if self.mode == "train" and self.aug:
            noise = np.zeros_like(data_2d)
            for j in range(self.config.n_joints):
                rnd_idx = np.random.randint(0, self.noise.shape[0], size=data_3d.shape[0])
                noise[:,j,:] = self.noise[rnd_idx,j,:]
            # noise = self.noise[rnd_idx, np.arange(self.config.n_joints), :]
            data_2d += noise

        data_2d = (data_2d - 0.5 - c) / f
        # flip the training data with a probability of 0.5
        flip = np.random.random() < 0.5
        if self.mode == "train" and flip:
            data_2d = data_2d.copy()
            data_2d[:, :, 0] *= -1
            data_2d = data_2d[:, self.left_right_symmetry_2d, :]  # [T,V,C]

            data_3d = data_3d.copy()
            data_3d[:, :, 0] *= -1
            data_3d = data_3d[:, self.left_right_symmetry_2d, :]

        if self.mode == "test":
            data_2d_flip = data_2d.copy()
            data_2d_flip[:, :, 0] *= -1
            data_2d_flip = data_2d_flip[:, self.left_right_symmetry_2d, :]
            data_2d_flip = (data_2d_flip - self.center_data_stat['mean']) / self.center_data_stat['std']

        data_2d = (data_2d - self.center_data_stat['mean']) / self.center_data_stat['std']

        data_2d = torch.from_numpy(data_2d.transpose((2, 0, 1))).float()  # [C,T,V]
        data_3d = torch.from_numpy(data_3d.transpose((2, 0, 1))).float()  # [C,T,V]

        # mean_3d = torch.from_numpy(self.mean_3d).float().unsqueeze(-1).unsqueeze(-1)  # [C,1,1]
        # std_3d = torch.from_numpy(self.std_3d).float().unsqueeze(-1).unsqueeze(-1)  # [C,1,1]
        mean_3d = torch.from_numpy(self.mean_3d).float().unsqueeze(-1)
        mean_3d = mean_3d.permute(1, 2, 0) if hasattr(self.config, 'PJN') and self.config.PJN else mean_3d.unsqueeze(-1)
        std_3d = torch.from_numpy(self.std_3d).float().unsqueeze(-1)
        std_3d = std_3d.permute(1, 2, 0) if hasattr(self.config, 'PJN') and self.config.PJN else std_3d.unsqueeze(-1)

        if self.mode == "test":
            # data_2d_flip = (data_2d_flip - self.mean_2d) / self.std_2d
            data_2d_flip = torch.from_numpy(data_2d_flip.transpose((2, 0, 1))).float()  # [C,T,V]
            # data_3d_flip = (data_3d_flip - self.mean_3d) / self.std_3d
            # data_3d_flip = torch.from_numpy(data_3d_flip.transpose((2, 0, 1))).float()  # [C,T,V]

        ret = {
            "data_2d": data_2d,
            "data_3d": data_3d,
            "mean_3d": mean_3d,
            "std_3d": std_3d,
            "idx": idx,
            "indices": index,
        }

        if self.mode == "test":
            ret["data_2d_flip"] = data_2d_flip
            # ret["data_3d_flip"] = data_3d_flip
        elif self.config.test_d2_type != 'data_2d_gt':
            ret["noise"] = noise

        return ret

    def get_valid_length(self):
        return self.valid_length


if __name__ == '__main__':
    from base.utilities import get_logger
    import torch
    from tqdm import tqdm


    class Empty:
        pass


    cfg = Empty()
    cfg.data_path = 'Workspace/Dataset/h36m_processed/structured_data'
    cfg.n_frames = 96
    cfg.n_joints = 17
    cfg.window_slide = 5
    cfg.test_d2_type = 'tcn_cpn'
    cfg.PJN = True
    # cfg.d2_type = 'tcn_cpn_20484g'

    logger = get_logger()
    train_data = H36M_Dataset(config=cfg, mode='train', logger=logger)
    val_data = H36M_Dataset(config=cfg, mode='test', logger=logger)

    noise = []
    for i in tqdm(range(len(train_data))):
        data = train_data.__getitem__(i)
        noise.append(data["noise"])
    error = np.concatenate(noise, axis=0)

    num_joint = cfg.n_joints
    name_joint = ['Hip', \
                  'RightHip', 'RightKnee', 'RightFoot', \
                  'LeftHip', 'LeftKnee', 'LeftFoot', \
                  'Spine', 'Neck', 'Head', 'Site', \
                  'LeftShoulder', 'LeftElbow', 'LeftHand', \
                  'RightShoulder', 'RightElbow', 'RightHand']

    # create save directory
    save_dir = 'vis_res/stat_mix_tcn_cpn_verify'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ----------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    for j in range(num_joint):
        e = error[:, j, :]
        m = train_data.noise_stat["mean"][j]
        s = train_data.noise_stat["std"][j]
        w = train_data.noise_stat["weight"][j]
        for i in range(2):
            pdf = lambda x: np.exp(-((x - m[i]) ** 2.0) / (2 * s[i] ** 2.0)) * w / np.sqrt(2 * np.pi * s[i] ** 2.0) + (
                    1 - w) / 100.0

            tick_x_val = [-20, -10, 0, 10, 20]
            tick_x_lab = ['-20', '-10', '0', '10', '20']

            plt.clf()
            plt.hist(e[:, i], bins=100, histtype='stepfilled', range=[-20.0, 20.0], density=True)

            x_sample = np.linspace(-20, 20, 1000)
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
