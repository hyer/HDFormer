from os.path import join
from pathlib import Path
import numpy as np
# import logging
import torch.utils.data

from dataset.lib import get_skeleton


class MPIINF3DHP_Dataset(torch.utils.data.Dataset):
    """
    dataset return instance: [C,T,V]
    dataloader return instance: [B,C,T,V]
    """

    def __init__(self, config, mode, logger):
        # self.logger = logging.getLogger(self.__class__.__name__)
        train_seqs = [
                         ('S1', 'Seq1'), ('S1', 'Seq2'), ('S2', 'Seq1'), ('S2', 'Seq2'), ('S3', 'Seq1'), ('S3', 'Seq2'),
                         ('S4', 'Seq2'),
                         ('S5', 'Seq1'), ('S5', 'Seq2'), ('S6', 'Seq1'), ('S6', 'Seq2'), ('S7', 'Seq1'), ('S7', 'Seq2'),
                         ('S8', 'Seq1')
                     ]
        vnect_cameras = [0, 1, 2, 4, 5, 6, 7, 8]

        self.logger = logger
        assert mode in ["train", "test"], "Invalid mode: {}".format(mode)
        home = str(Path.home())
        config.data_path = str(join(home, config.data_path)) + '/'
        self.config = config
        self.mode = mode
        self.skeleton = get_skeleton()
        train_path = config.data_path + "train.npz"
        test_path = config.data_path + "test.npz"
        f = np.load(train_path, allow_pickle=True)
        train_data, train_stat = f['data'].item(), f['center_data_stat'].item()
        f = np.load(test_path, allow_pickle=True)
        test_data, test_stat = f['data'].item(), f['center_data_stat'].item()

        data = train_data if mode == 'train' else test_data
        stat = test_stat if config.use_test_stat else train_stat

        self.n_frames = config.n_frames
        self.n_joints = config.n_joints
        self.window_slide = config.window_slide

        self.left_right_symmetry_2d = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        # self.left_right_symmetry_3d = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12])

        self.data_2d = {}
        # self.data_2d_centered = {}
        self.data_3d = {}
        self.cameras = {}
        self.indices = []
        self.valid_index = {} if mode == 'test' else None

        # cut videos into short clips of fixed length
        self.logger.info("Loading sequence...")
        for idx, k in enumerate(data):
            # import pdb
            # pdb.set_trace()
            if mode == 'train' and k not in train_seqs:
                continue
            # d3 = data[k]['annot3']  # [Cam,T,V,C]
            # [Cam,T,V,C]
            d3 = data[k]['univ_annot3'] if hasattr(config, 'univ_annot3') and config.univ_annot3 else data[k]['annot3']
            # align root to origin
            d3 = d3 - d3[:, :, :1, :]
            # remove zero root joint
            # d3 = d3[:, 1:, :]
            self.data_3d[k] = d3
            d2 = data[k]['annot2']  # [Cam,T,V,C]
            self.data_2d[k] = d2
            self.cameras[k] = data[k]['cams']  # [Cam,4]
            if self.valid_index is not None:
                self.valid_index[k] = data[k]['valid_idx']

            Cams, N = d3.shape[0], d3.shape[1]
            for cam in range(Cams):
                if mode == 'train' and cam not in vnect_cameras:
                    continue
                n = 0
                while n + self.n_frames <= N:
                    self.indices.append((idx,) + (k, cam) + (n, self.n_frames))
                    n += self.window_slide
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
            data_stat = np.concatenate(list(self.data_3d.values()), axis=1)  # take care joint 0
            self.mean_3d = np.mean(data_stat, axis=(0, 1))
            self.std_3d = np.std(data_stat, axis=(0, 1))
            stats_data = self.mean_3d, self.std_3d
            config.states = stats_data
            self.logger.info("Saving stats...")

        self.mean_2d, self.std_2d = stat['mean'], stat['std']
        self.logger.info("mean 3d: {}".format(self.mean_3d))
        self.logger.info("std 3d: {}".format(self.std_3d))
        self.logger.info("mean 2d: {}".format(self.mean_2d))
        self.logger.info("std 2d: {}".format(self.std_2d))

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        assert 0 <= item < self.n_data, "Index {} out of range [{}, {})".format(item, 0, self.n_data)

        index = self.indices[item]
        idx, k, cam_idx, n, t = index
        # if self.mode == 'test':
        #     if k in ['TS1', 'TS2', 'TS3', 'TS4']:
        #         f = np.array([7.32506, 7.32506]) * 2048. / 10.
        #         c = np.array([-0.0322884, 0.0929296]) * 2048. / 10. + np.array([1024, 1024])
        #     else:
        #         f = np.array([8.770747185, 8.770747185]) * np.array([1920 / 10.000000000, 1080 / 5.625000000])
        #         c = np.array([-0.104908645, 0.104899704]) * np.array([1920 / 10.000000000, 1080 / 5.625000000]) \
        #             + np.array([960, 540])
        # else:
        cam = self.cameras[k][cam_idx, :]
        f, c = cam[-2:], cam[:2]
        data_3d = self.data_3d[k][cam_idx, n: n + t, ...].copy()
        data_2d = self.data_2d[k][cam_idx, n: n + t, ...].copy()
        data_2d = (data_2d - c) / f
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
            data_2d_flip = (data_2d_flip - self.mean_2d) / self.std_2d

        data_2d = (data_2d - self.mean_2d) / self.std_2d

        data_2d = torch.from_numpy(data_2d.transpose((2, 0, 1))).float()  # [C,T,V]
        data_3d = torch.from_numpy(data_3d.transpose((2, 0, 1))).float()  # [C,T,V]

        mean_3d = torch.from_numpy(self.mean_3d).float().unsqueeze(-1)
        mean_3d = mean_3d.permute(1, 2, 0) if hasattr(self.config, 'PJN') and self.config.PJN else mean_3d.unsqueeze(-1)
        std_3d = torch.from_numpy(self.std_3d).float().unsqueeze(-1)
        std_3d = std_3d.permute(1, 2, 0) if hasattr(self.config, 'PJN') and self.config.PJN else std_3d.unsqueeze(-1)

        if self.mode == "test":
            data_2d_flip = torch.from_numpy(data_2d_flip.transpose((2, 0, 1))).float()  # [C,T,V]

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

        return ret


if __name__ == '__main__':
    from base.utilities import get_logger
    import torch
    from tqdm import tqdm


    class Empty:
        pass


    cfg = Empty()
    cfg.data_path = 'Workspace/Dataset/mpi_inf_3dhp'
    cfg.n_frames = 96
    cfg.n_joints = 17
    cfg.window_slide = 5
    cfg.test_d2_type = 'data_2d_gt'
    cfg.PJN = True
    cfg.use_test_stat = True
    # cfg.d2_type = 'tcn_cpn_20484g'

    logger = get_logger()
    train_data = MPIINF3DHP_Dataset(config=cfg, mode='train', logger=logger)
    val_data = MPIINF3DHP_Dataset(config=cfg, mode='test', logger=logger)
