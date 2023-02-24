import torch
import torch.nn as nn
from metrics.dct import LinearDCT
import torch.nn.functional as F


class TrajectoryDCTLoss(nn.Module):
    def __init__(self, win_size=24, shape='BCTV'):
        super().__init__()
        # assert frames // win_size == 0, "Unsuitable win_size!"
        self.win_size = win_size
        self.shape = shape

        self.dct = LinearDCT(win_size, 'dct', norm='ortho')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.shape == 'BCTV':
            # B,C,V,num_win,win_size
            pred_reshape = pred.permute(0, 1, 3, 2).contiguous().unfold(dimension=-1, size=self.win_size,
                                                                        step=self.win_size)
            gt_reshape = gt.permute(0, 1, 3, 2).contiguous().unfold(dimension=-1, size=self.win_size,
                                                                    step=self.win_size)
            #
            pred_reshape = pred_reshape.contiguous().view(-1, self.win_size)
            gt_reshape = gt_reshape.contiguous().view(-1, self.win_size)
        elif self.shape == 'TVC':
            pred_reshape = pred.permute(1, 2, 0).contiguous().unfold(-1, self.win_size,
                                                                     self.win_size).contiguous().view(-1, self.win_size)
            gt_reshape = gt.permute(1, 2, 0).contiguous().unfold(-1, self.win_size,
                                                                 self.win_size).contiguous().view(-1, self.win_size)
        else:
            raise NotImplementedError
        pred_dct, gt_dct = self.dct(pred_reshape), self.dct(gt_reshape)
        return torch.mean(torch.norm(pred_dct - gt_dct, dim=-1))


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0).unsqueeze(-1)


class SSIM(torch.nn.Module):
    def __init__(self, channel=3, windowSize=11, reduceMean=True, transform=None, cuda=False, reverse=False):
        super(SSIM, self).__init__()
        self.reverse = reverse
        self.transform = transform
        self.windowSize = windowSize
        self.reduceMean = reduceMean
        self.channel = channel
        self.kernel = _fspecial_gauss_1d(windowSize, sigma=1.5)
        self.kernel.requires_grad = False
        if self.channel > 1:
            self.kernel = torch.cat([self.kernel for _ in range(self.channel)], dim=0)
        if cuda:
            self.kernel = self.kernel.cuda()

    def forward(self, signal1, signal2):
        """
        :param signal1: B*C*T*V cuda tensor
        :param signal2: B*C*T*V cuda tensor
        :return: ssim or ssim map
        """
        if self.transform is not None:
            signal1 = self.transform(signal1)
            signal2 = self.transform(signal2)

        mu1 = F.conv2d(signal1, self.kernel, groups=self.channel)
        mu2 = F.conv2d(signal2, self.kernel, groups=self.channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(signal1 * signal1, self.kernel, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(signal2 * signal2, self.kernel, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(signal1 * signal2, self.kernel, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduceMean:
            return ssim_map.mean() if not self.reverse else 1 - ssim_map.mean()
        else:
            return ssim_map.mean(dim=1, keepdim=True) if not self.reverse else 1 - ssim_map.mean(dim=1, keepdim=True)
