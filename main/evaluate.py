import tqdm
import numpy as np
from os.path import join
from mmcv.utils import get_logger
from dataset.lib import get_skeleton
from models.lib.directed_graph import DiGraph
import torch
from metrics import *


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    import numpy as np

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def h36m_evaluate(preds_3d, gts_3d, indices, test_dataset, config):
    """
    Evaluate on Human3.6M dataset. Action-wise and overall errors are measured.
    """
    logger = get_logger(name='h36m_evaluate')
    logger.info("Evaluating on videos...")

    all_frames_preds = []
    all_frames_gts = []
    all_frames_indices = []

    current_video_id = -1
    current_frame_id = -1

    n_sample = gts_3d.shape[0]

    video_start_indices = []

    index_to_action = {}
    action_preds = {}
    action_gts = {}
    for idx, k in enumerate(sorted(test_dataset.data_3d)):
        index_to_action[idx] = k[1]
        if k[1] not in action_preds:
            action_preds[k[1]] = []
            action_gts[k[1]] = []

    for i in tqdm.tqdm(range(n_sample)):
        if indices[i] != current_video_id:
            # start in a new video
            current_video_id = indices[i]
            current_frame_id = 0
            video_start_indices.append(len(all_frames_preds))

        for t in range(0 if current_frame_id == 0 else config.n_frames - config.window_slide, config.n_frames):
            frm = []
            for j in range(t // config.window_slide + 1):
                if i + j >= n_sample or indices[i + j] != current_video_id:
                    break
                # collect estimations from multiple samples which have overlapping at the current frame t
                frm.append(preds_3d[i + j, :, :, t - j * config.window_slide])
            frm = np.array(frm)
            frm = np.mean(frm, axis=0)
            all_frames_preds.append(frm)
            all_frames_gts.append(gts_3d[i, :, :, t])
            all_frames_indices.append(indices[i])

            current_frame_id += 1

    all_frames_preds = np.array(all_frames_preds)
    all_frames_gts = np.array(all_frames_gts)
    all_frames_indices = np.array(all_frames_indices)

    # add back the root joints
    # import pdb
    # pdb.set_trace()
    # all_frames_preds = np.concatenate([np.zeros([all_frames_preds.shape[0], 1, 3]), all_frames_preds], axis=1)
    # all_frames_gts = np.concatenate([np.zeros([all_frames_gts.shape[0], 1, 3]), all_frames_gts], axis=1)

    for idx, start in enumerate(video_start_indices):
        if idx + 1 == len(video_start_indices):
            cp = all_frames_preds[start:]
            cg = all_frames_gts[start:]
            ci = all_frames_indices[start:]
        else:
            cp = all_frames_preds[start:video_start_indices[idx + 1]]
            cg = all_frames_gts[start:video_start_indices[idx + 1]]
            ci = all_frames_indices[start:video_start_indices[idx + 1]]
        assert ci[0] == ci[-1]
        ci = ci[0]
        action_preds[index_to_action[ci]].append(cp)
        action_gts[index_to_action[ci]].append(cg)

    # save
    np.savez_compressed(join(config.save_folder, 'pred.npz'), data=action_preds)
    np.savez_compressed(join(config.save_folder, 'gt.npz'), data=action_gts)

    allp = []
    allg = []
    
    # Average
    avg_res = {}
    for act in sorted(action_preds):
        lp = np.concatenate(action_preds[act], axis=0)
        lg = np.concatenate(action_gts[act], axis=0)
        allp.append(lp)
        allg.append(lg)

        res = error(lp, lg, config)
        for k, v in res.items():
            if k not in avg_res:
                avg_res[k] = v / len(action_preds) # len(action_preds) = 15
            else:
                avg_res[k] += v / len(action_preds)
        print("{:15s} {:>6d} frames".format(act, lp.shape[0]), res)

    print("Average:", avg_res)

    allp = np.concatenate(allp, axis=0)
    allg = np.concatenate(allg, axis=0)
    res = error(allp, allg, config)
    print("{:15s} {:>6d} frames".format("All", allp.shape[0]), res)


def pck(preds, gts, threshold=150):
    # import pdb
    # pdb.set_trace()
    # included_joints = [10, 8, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6]
    groups = [[10], [8], [11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]]
    dists = np.sqrt(np.sum(np.square(preds - gts), axis=2))
    err = []
    for g in groups:
        if len(g) > 1:
            e = np.mean(dists[:, g], axis=1, keepdims=False)
        else:
            e = dists[:, g[0]]
        err.append(e)
    # dists = dists[:, included_joints]
    dists = np.stack(err, axis=-1)
    assert dists.shape[-1] == 8, dists.shape
    return np.mean((dists <= threshold))


def auc(preds, gts, threshold_interval=5):
    res = []
    for inter in range(0, 150 + threshold_interval, threshold_interval):
        res.append(pck(preds, gts, inter))
    return np.array(res)


def mpii_joint_groups():
    joint_groups = [
        ['Head', [10]],
        ['Neck', [8]],
        ['Shou', [11, 14]],
        ['Elbow', [12, 15]],
        ['Wrist', [13, 16]],
        ['Hip', [1, 4]],
        ['Knee', [2, 5]],
        ['Ankle', [3, 6]],
    ]
    all_joints = []
    for i in joint_groups:
        all_joints += i[1]
    return joint_groups, all_joints


def mean(l):
    return sum(l) / len(l)


def mpii_compute_3d_pck(seq_err):
    pck_curve_array = []
    pck_array = []
    auc_array = []
    thresh = np.arange(0, 151, 5)
    pck_thresh = 150
    joint_groups, all_joints = mpii_joint_groups()
    for seq_idx in range(len(seq_err)):
        pck_curve = []
        pck_seq = []
        auc_seq = []
        err = np.array(seq_err[seq_idx]).astype(np.float32)

        for j in range(len(joint_groups)):
            err_selected = err[:, joint_groups[j][1]]
            buff = []
            for t in thresh:
                pck = np.float32(err_selected < t).sum() / len(joint_groups[j][1]) / len(err)
                buff.append(pck)  # [Num_thresholds]
            pck_curve.append(buff)
            auc_seq.append(mean(buff))
            pck = np.float32(err_selected < pck_thresh).sum() / len(joint_groups[j][1]) / len(err)
            pck_seq.append(pck)

        buff = []
        for t in thresh:
            pck = np.float32(err[:, all_joints] < t).sum() / len(err) / len(all_joints)
            buff.append(pck)  # [Num_thresholds]
        pck_curve.append(buff)

        pck = np.float32(err[:, all_joints] < pck_thresh).sum() / len(err) / len(all_joints)
        pck_seq.append(pck)

        pck_curve_array.append(pck_curve)  # [num_seq: [Num_grpups+1: [Num_thresholds]]]
        pck_array.append(pck_seq)  # [num_seq: [Num_grpups+1]]
        auc_array.append(auc_seq)  # [num_seq: [Num_grpups]]

    # return pck_curve_array, pck_array, auc_array
    pck = [mean(x[:-1]) for x in pck_array]
    pck = mean(pck)
    auc = [mean(x) for x in auc_array]
    auc = mean(auc)
    return pck, auc


def error(preds, gts, config):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    shape: NVC
    """
    N = preds.shape[0]
    skeleton = get_skeleton()
    diG = DiGraph(skeleton)
    # trajectory_dct = TrajectoryDCTLoss(config.dct_win_size, shape='TVC')
    # trajectory_ssim = SSIM(windowSize=config.ssim_win_size, cuda=False)
    if hasattr(config, 'traj_loss'):
        traj_fn = eval(config.traj_loss)

    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    pampjpe = np.zeros([N, config.n_joints])
    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)

    # bone_sym_err
    left_parent = preds[:, skeleton.parents()[skeleton._joints_left], :]
    left = preds[:, skeleton._joints_left, :]
    right_parent = preds[:, skeleton.parents()[skeleton._joints_right], :]
    right = preds[:, skeleton._joints_right, :]
    left_bone_lengths = np.linalg.norm(left - left_parent, axis=-1, keepdims=True)
    right_bone_lengths = np.linalg.norm(right - right_parent, axis=-1, keepdims=True)
    bone_sym_err = np.mean(np.abs(left_bone_lengths - right_bone_lengths))

    # bone
    pre_bone = (preds[:, [c for p, c in diG.directed_edges_hop1], :] - preds[:, [p for p, c in diG.directed_edges_hop1], :])
    gt_bone = (gts[:, [c for p, c in diG.directed_edges_hop1], :] - gts[:, [p for p, c in diG.directed_edges_hop1], :])
    bone_err = np.mean(
        np.abs(np.linalg.norm(pre_bone, axis=-1, keepdims=True) - np.linalg.norm(gt_bone, axis=-1, keepdims=True)))
    direc_err = torch.cosine_similarity(torch.from_numpy(pre_bone), torch.from_numpy(gt_bone), dim=-1).numpy()
    direc_err = np.mean(np.arccos(direc_err.clip(-1, 1))) / np.pi * 180.0

    # trajectory
    # trajectory_dct_err = trajectory_dct(torch.from_numpy(preds), torch.from_numpy(gts)).item()
    # trajectory_ssim_err = trajectory_ssim(
    #     torch.from_numpy(preds).permute(2, 0, 1).unsqueeze(0),
    #     torch.from_numpy(gts).permute(2, 0, 1).unsqueeze(0)
    # ).item()
    traj_err = traj_fn(
        torch.from_numpy(preds).permute(2, 0, 1).unsqueeze(0).cuda(),
        torch.from_numpy(gts).permute(2, 0, 1).unsqueeze(0).cuda()
    ).item()

    # PCK
    # import pdb
    # pdb.set_trace()
    # pck_val = pck(preds, gts)
    # auc_val = auc(preds, gts, threshold_interval=5)

    return {"mpjpe": mpjpe, "pampjpe": pampjpe, "bone_sym_err": bone_sym_err, "bone_err": bone_err,
            "direc_err": direc_err, "traj_err": traj_err}


def MPIINF3DHP_evaluate(preds_3d, gts_3d, indices, test_dataset, config):
    """
    Evaluate on Human3.6M dataset. Action-wise and overall errors are measured.
    """
    logger = get_logger(name='MPIINF3DHP_evaluate')
    logger.info("Evaluating on videos...")

    all_frames_preds = []
    all_frames_gts = []
    all_frames_indices = []

    current_video_id = -1
    current_frame_id = -1

    n_sample = gts_3d.shape[0]

    video_start_indices = []

    index_to_squence = {}
    sq_preds = {}
    sq_gts = {}
    for idx, k in enumerate(test_dataset.data_3d):
        index_to_squence[idx] = k
        if k not in sq_preds:
            sq_preds[k] = []
            sq_gts[k] = []

    for i in tqdm.tqdm(range(n_sample)):
        if indices[i] != current_video_id:
            # start in a new video
            current_video_id = indices[i]
            current_frame_id = 0
            video_start_indices.append(len(all_frames_preds))

        for t in range(0 if current_frame_id == 0 else config.n_frames - config.window_slide, config.n_frames):
            frm = []
            for j in range(t // config.window_slide + 1):
                if i + j >= n_sample or indices[i + j] != current_video_id:
                    break
                # collect estimations from multiple samples which have overlapping at the current frame t
                frm.append(preds_3d[i + j, :, :, t - j * config.window_slide])
            frm = np.array(frm)
            frm = np.mean(frm, axis=0)
            all_frames_preds.append(frm)
            all_frames_gts.append(gts_3d[i, :, :, t])
            all_frames_indices.append(indices[i])

            current_frame_id += 1

    all_frames_preds = np.array(all_frames_preds)
    all_frames_gts = np.array(all_frames_gts)
    all_frames_indices = np.array(all_frames_indices)

    # add back the root joints
    # import pdb
    # pdb.set_trace()
    # all_frames_preds = np.concatenate([np.zeros([all_frames_preds.shape[0], 1, 3]), all_frames_preds], axis=1)
    # all_frames_gts = np.concatenate([np.zeros([all_frames_gts.shape[0], 1, 3]), all_frames_gts], axis=1)

    for idx, start in enumerate(video_start_indices):
        if idx + 1 == len(video_start_indices):
            cp = all_frames_preds[start:]
            cg = all_frames_gts[start:]
            ci = all_frames_indices[start:]
        else:
            cp = all_frames_preds[start:video_start_indices[idx + 1]]
            cg = all_frames_gts[start:video_start_indices[idx + 1]]
            ci = all_frames_indices[start:video_start_indices[idx + 1]]
        assert ci[0] == ci[-1]
        ci = ci[0]
        sq_preds[index_to_squence[ci]].append(cp)
        sq_gts[index_to_squence[ci]].append(cg)

    # save
    np.savez_compressed(join(config.save_folder, 'pred.npz'), data=sq_preds)
    np.savez_compressed(join(config.save_folder, 'gt.npz'), data=sq_gts)

    allp = []
    allg = []
    # import pdb
    # pdb.set_trace()
    seq_err = []
    for act in sorted(sq_preds):
        valid_idx = test_dataset.valid_index[act]
        lp = np.concatenate(sq_preds[act], axis=0)
        lg = np.concatenate(sq_gts[act], axis=0)
        lp, lg = (lp[valid_idx], lg[valid_idx]) if hasattr(config, 'use_valid_frames') and config.use_valid_frames \
            else (lp, lg)
        allp.append(lp)
        allg.append(lg)
        seq_err.append(np.sqrt(np.power(lp - lg, 2).sum(axis=-1)))
        res = error(lp, lg, config)
        print("{:15s} {:>6d} frames".format(act, lp.shape[0]), res)

    allp = np.concatenate(allp, axis=0)
    allg = np.concatenate(allg, axis=0)
    res = error(allp, allg, config)
    # import pdb
    # pdb.set_trace()
    pck_array, auc_array = mpii_compute_3d_pck(seq_err)
    print("PCK", pck_array)
    print("AUC", auc_array)
    print("{:15s} {:>6d} frames".format("All", allp.shape[0]), res)
