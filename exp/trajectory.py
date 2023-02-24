import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

t = np.array(range(101))
sig_gt = np.sin(np.pi * t / 50) * 0.3

sig1 = sig_gt + (np.random.rand(101) - 0.5) * 0.365
sig2 = np.sin(np.pi * t / 50) * 0.3 + np.sin(np.pi * t / 25) * 0.144
sig3 = np.sin(np.pi * t / 50) * 0.3 + np.cos(np.pi * t / 100) * 0.142

# plt.figure(figsize=(20, 2))
# plt.plot(sig_gt, '.-')
# # plt.plot(sig1, 'r.-')
# plt.plot(sig2, 'g.-')
# # plt.plot(sig3, '.-')
# plt.show()
diff1 = np.sqrt(np.mean((sig_gt - sig1) ** 2))
diff2 = np.sqrt(np.mean((sig_gt - sig2) ** 2))
diff3 = np.sqrt(np.mean((sig_gt - sig3) ** 2))
print(diff1, diff2, diff3)

from metrics.tracjectory_loss import TrajectoryDCTLoss, SSIM
import torch

sig_gt = torch.from_numpy(sig_gt).float().unsqueeze(-1).unsqueeze(-1)
sig1 = torch.from_numpy(sig1).float().unsqueeze(-1).unsqueeze(-1)
sig2 = torch.from_numpy(sig2).float().unsqueeze(-1).unsqueeze(-1)
sig3 = torch.from_numpy(sig3).float().unsqueeze(-1).unsqueeze(-1)

diff_dct = TrajectoryDCTLoss(win_size=100, shape='TVC')
dct_diff1 = diff_dct(sig_gt, sig1)
dct_diff2 = diff_dct(sig_gt, sig2)
dct_diff3 = diff_dct(sig_gt, sig3)
print(dct_diff1, dct_diff2, dct_diff3)

sig_gt = sig_gt.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
sig1 = sig1.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
sig2 = sig2.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
sig3 = sig3.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
diff_ssim = SSIM(windowSize=11, channel=1)
ssim_diff1 = diff_ssim(sig_gt, sig1)
ssim_diff2 = diff_ssim(sig_gt, sig2)
ssim_diff3 = diff_ssim(sig_gt, sig3)
print(ssim_diff1, ssim_diff2, ssim_diff3)
