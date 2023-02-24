import torch


def avg_flip(pre, pre_flip):
    left_right_symmetry = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    pre_flip[:, 0, :, :] *= -1
    pre_flip = pre_flip[:, :, :, left_right_symmetry]
    pred_avg = (pre + pre_flip) / 2.
    return pred_avg


def avg_flip_2d(pre, pre_flip):
    left_right_symmetry = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
    pre_flip[:, 0, :, :] = 1001.0 - pre_flip[:, 0, :, :]
    pre_flip = pre_flip[:, :, :, left_right_symmetry]
    pred_avg = (pre + pre_flip) / 2.
    return pred_avg


def get_edge_fea(x_v, di_graph):
    return (x_v[..., [c for p, c in di_graph.directed_edges]] - x_v[
        ..., [p for p, c in di_graph.directed_edges]]).contiguous()


def reduce_lr(x, di_graph):
    """
    x: shape [BCV]
    """
    pool_lr = x[..., di_graph.edge_left] * 0.5 + x[..., di_graph.edge_right] * 0.5
    pool_lr = torch.cat((x[..., di_graph.edge_middle], pool_lr),
                        dim=-1).contiguous()
    return pool_lr


def unpool_lr(x: torch.Tensor, di_graph):
    """
    x: [B,1,1,10]
    """
    B, _, _, _ = x.shape
    E = di_graph.num_edges
    res = torch.ones(size=(B, 1, 1, E), dtype=x.dtype, device=x.device)
    res[..., di_graph.edge_middle] = x[..., :len(di_graph.edge_middle)]
    res[..., di_graph.edge_left] = x[..., len(di_graph.edge_middle):]
    res[..., di_graph.edge_right] = x[..., len(di_graph.edge_middle):]
    return res
