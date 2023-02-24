import torch
import torch.nn as nn

from base.base_model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super(ConvBlock, self).__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=pad * 4, dilation=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=pad, bias=False)

    def forward(self, x):
        y = self.residual(x)
        y += self.conv(x)
        return y


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super(TransConvBlock, self).__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=pad, output_padding=1,
                               bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=pad, output_padding=0,
                               bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.residual = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=pad,
                                           output_padding=1, bias=False)

    def forward(self, x):
        y = self.residual(x)
        y += self.conv(x)
        return y


class TCN_2D(BaseModel):
    def __init__(self, cfg):
        super(TCN_2D, self).__init__()
        pad = cfg.kernel_size // 2
        self.in_conv = nn.Sequential(
            nn.BatchNorm1d(cfg.n_joints * 2),
            nn.Conv1d(cfg.n_joints * 2, cfg.planes, cfg.kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(cfg.planes),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout)
        )
        self.conv_block1 = ConvBlock(cfg.planes, cfg.planes, cfg.kernel_size, cfg.dropout)
        self.conv_block2 = ConvBlock(cfg.planes, cfg.planes, cfg.kernel_size, cfg.dropout)
        self.trans_conv_block1 = TransConvBlock(cfg.planes, cfg.planes, cfg.kernel_size, cfg.dropout)
        self.trans_conv_block2 = TransConvBlock(cfg.planes, cfg.planes, cfg.kernel_size, cfg.dropout)
        self.out_conv = nn.Sequential(
            nn.Conv1d(cfg.planes, cfg.planes, cfg.kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(cfg.planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfg.planes, cfg.n_joints * 2, kernel_size=1, bias=True)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         # nn.init.normal_(m.weight.data)
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             # nn.init.normal_(m.bias.data)
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     # elif isinstance(m, nn.ConvTranspose1d):
        #     #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     #     nn.init.constant_(m.bias, 0)

    def forward(self, v: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
            x: shape [B,C,T,V_v]
        """
        # return v * std_2d_gt + mean_2d_gt
        B, C, T, V = v.shape
        x = v.permute(0, 1, 3, 2).contiguous().view(B, -1, T).contiguous()  # B,C*V,T
        x = self.in_conv(x)
        x1 = self.conv_block1(x)
        x = self.conv_block2(x1)
        x = self.trans_conv_block1(x)
        x += x1
        x = self.trans_conv_block2(x)
        # import pdb
        # pdb.set_trace()
        x = self.out_conv(x)
        pre_joints = x.view(B, 2, V, T).permute(0, 1, 3, 2).contiguous()  # [B,2,T,V]
        pre_joints += (v*std + mean)
        # pre_joints = pre_joints * std + mean
        return pre_joints
