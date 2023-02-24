from os.path import join
from pathlib import Path
import numpy as np
import h5py
from base.utilities import get_logger
from scipy.io import loadmat


class Annotations:
    def __init__(self, annot):
        self.annot = annot
        self.annot3 = self._reshape_annot(annot['annot3'], 3)
        self.univ_annot3 = self._reshape_annot(annot['univ_annot3'], 3)
        self.annot2 = self._reshape_annot(annot['annot2'], 2)
        self.cams = infer_cam(self.annot3, self.annot2)

    @staticmethod
    def _reshape_annot(arr, ndims):
        arr = np.stack(arr.flatten())
        return arr.reshape((arr.shape[0], arr.shape[1], 28, ndims))


def infer_cam(annot3All, annot2All):
    cams = []
    for annot3, annot2 in zip(annot3All, annot2All):
        x3d = np.stack([annot3[:, :, 0], annot3[:, :, 2]], axis=-1).reshape(-1, 2)
        x2d = (annot2[:, :, 0] * annot3[:, :, 2]).reshape(-1, 1)
        fx, cx = list(np.linalg.lstsq(x3d, x2d, rcond=None)[0].flatten())
        y3d = np.stack([annot3[:, :, 1], annot3[:, :, 2]], axis=-1).reshape(-1, 2)
        y2d = (annot2[:, :, 1] * annot3[:, :, 2]).reshape(-1, 1)
        fy, cy = list(np.linalg.lstsq(y3d, y2d, rcond=None)[0].flatten())
        cams.append(np.array([cx, cy, fx, fy]))
    return np.stack(cams, axis=0)


def preprocess_train_data(src_dir):
    # skeleton = get_skeleton()
    index2h36m = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]
    dataset = {}
    centered_2d = {}
    center_data_stat = {}
    for TS in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
        for S in ['Seq1', 'Seq2']:
            annot = Annotations(loadmat(join(src_dir, TS, S, 'annot.mat')))
            dataset[(TS, S)] = {'annot3': annot.annot3[..., index2h36m, :],
                                'univ_annot3': annot.univ_annot3[..., index2h36m, :],
                                'annot2': annot.annot2[..., index2h36m, :],
                                'cams': annot.cams}
            centered_2d[(TS, S)] = (dataset[(TS, S)]['annot2'] - np.expand_dims(annot.cams[:, :2],
                                                                                (1, 2))) / np.expand_dims(
                annot.cams[:, -2:], (1, 2))
    c_data = np.concatenate(list(centered_2d.values()), axis=1)
    center_data_stat['mean'] = np.mean(c_data, axis=(0, 1))
    center_data_stat['std'] = np.std(c_data, axis=(0, 1))
    np.savez_compressed(join(src_dir, 'train.npz'), data=dataset, center_data_stat=center_data_stat)


def preprocess_test_data(src_dir):
    # skeleton = get_skeleton()
    index2h36m = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]
    dataset = {}
    centered_2d = {}
    center_data_stat = {}
    for TS in ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']:
        with h5py.File(join(src_dir, TS, 'annot_data.mat'), 'r') as annot:
            indices = []
            for frame_index, is_valid in enumerate(np.array(annot['valid_frame']).flatten()):
                if is_valid == 1:
                    indices.append(frame_index)

            n_frames = len(annot['annot3'])
            annot3 = np.array(annot['annot3']).reshape(1, n_frames, 17, 3)
            univ_annot3 = np.array(annot['univ_annot3']).reshape(1, n_frames, 17, 3)
            annot2 = np.array(annot['annot2']).reshape(1, n_frames, 17, 2)

            annot3, univ_annot3, annot2 = annot3[:, :, index2h36m, :], univ_annot3[:, :, index2h36m, :], annot2[:, :,
                                                                                                         index2h36m, :]

            cams = infer_cam(annot3, annot2)
            centered_2d[TS] = (annot2 - np.expand_dims(cams[:, :2], (1, 2))) / np.expand_dims(
                cams[:, -2:], (1, 2))
            dataset[TS] = {'annot3': annot3, 'univ_annot3': univ_annot3, 'annot2': annot2, 'valid_idx': indices,
                           'cams': cams}
    c_data = np.concatenate(list(centered_2d.values()), axis=1)
    center_data_stat['mean'] = np.mean(c_data, axis=(0, 1))
    center_data_stat['std'] = np.std(c_data, axis=(0, 1))
    np.savez_compressed(join(src_dir, 'test.npz'), data=dataset, center_data_stat=center_data_stat)


if __name__ == '__main__':
    logger = get_logger()
    home = str(Path.home())
    base_path = join(home, 'Workspace/Dataset/mpi_inf_3dhp')
    preprocess_train_data(base_path)
    base_path = join(home, 'Workspace/Dataset/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set')
    preprocess_test_data(base_path)
