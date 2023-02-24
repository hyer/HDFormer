import torch
import torch.nn as nn
from base.base_model import BaseModel
from models.hd_former import *


class Model(BaseModel):

    def __init__(self, skeleton, cfg):
        super(Model, self).__init__()
        self.regress_with_edge = hasattr(cfg, 'regress_with_edge') and cfg.regress_with_edge
        self.backbone = eval(cfg.backbone)(skeleton, cfg)
        num_v, num_e = self.backbone.di_graph.source_M.shape
        self.regressor_type = cfg.regressor_type if hasattr(cfg, 'regressor_type') else 'conv'
        if self.regressor_type == 'conv':
            self.joint_regressor = nn.Conv2d(self.backbone.PLANES[0], 3 * (num_v - 1),
                                             kernel_size=(3, num_v + num_e) if self.regress_with_edge else (3, num_v),
                                             padding=(1, 0), bias=True)
        elif self.regressor_type == 'fc':
            self.joint_regressor = nn.Conv1d(
                self.backbone.PLANES[0] * (num_v + num_e) if self.regress_with_edge else
                self.backbone.PLANES[0] * num_v,
                3 * (num_v - 1),
                kernel_size=3,
                padding=1, bias=True)
        else:
            raise NotImplemented

    def forward(self, x_v: torch.Tensor, mean_3d: torch.Tensor, std_3d: torch.Tensor):
        """
        x: shape [B,C,T,V_v]
        """
        fv, fe = self.backbone(x_v)
        B, C, T, V = fv.shape
        _, _, _, E = fe.shape

        # import pdb
        # pdb.set_trace()
        # [B,3*(V-1),T,1]
        if self.regressor_type == 'conv':
            pre_joints = self.joint_regressor(
                torch.cat([fv, fe], dim=-1)) if self.regress_with_edge else self.joint_regressor(fv)
        elif self.regressor_type == 'fc':
            x = (torch.cat([fv, fe], dim=-1) if self.regress_with_edge else fv) \
                .permute(0, 1, 3, 2).contiguous().view(B, -1, T)
            pre_joints = self.joint_regressor(x)
        else:
            raise NotImplemented
        pre_joints = pre_joints.view(B, 3, V - 1, T).permute(0, 1, 3, 2).contiguous()  # [B,3,T,V-1]
        pre_joints = torch.cat(
            (torch.zeros((B, 3, T, 1), dtype=pre_joints.dtype, device=pre_joints.device),
             pre_joints),
            dim=-1)
        pre_joints = pre_joints * std_3d + mean_3d
        return pre_joints
