import os
import sys

from fontTools.unicodedata import script
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ThreeDCCN import Cylindrical_Net
from .common import *



class Descriptor_Net(nn.Module):
    def __init__(self,outdim):
        super(Descriptor_Net, self).__init__()
        self.rad_n = 9
        self.azi_n = 60
        self.ele_n = 30
        self.des_r = 2.0
        self.voxel_r = 0.3
        self.voxel_sample = 30
        self.dataset = "dataset"



        self.bn_xyz_raising = nn.BatchNorm2d(16)
        self.bn_mapping = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.xyz_raising = nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv_net = Cylindrical_Net(inchan=16, dim=outdim)
        # 将模型移到GPU
        self.cuda()

    def forward(self, input):
        center = input[:, -1, :].unsqueeze(1)
        delta_x = input[:, :, 0:3] - center[:, :, 0:3]  # (B, npoint, 3), normalized coordinates
        # for case in switch(self.dataset):
        #     if case('3DMatch'):
        #         z_axis = cal_Z_axis(delta_x, ref_point=input[:, -1, :3])
        #         z_axis = l2_norm(z_axis, axis=1)
        #         R = RodsRotatFormula(z_axis, torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(z_axis.shape[0], 1))
        #         delta_x = torch.matmul(delta_x, R)
        #         break
        #     if case('KITTI'):
        #         break

        # del
        z_axis = cal_Z_axis(delta_x, ref_point=input[:, -1, :3])
        z_axis = l2_norm(z_axis, axis=1)
        R = RodsRotatFormula(z_axis, torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(z_axis.shape[0], 1))
        delta_x = torch.matmul(delta_x, R)

        # partition the local surface along elevator, azimuth, radial dimensions
        S2_xyz = torch.FloatTensor(get_voxel_coordinate(radius=self.des_r,
                                                           rad_n=self.rad_n,
                                                           azi_n=self.azi_n,
                                                           ele_n=self.ele_n))

        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()
        # query points in sphere
        new_points = sphere_query(delta_x, pts_xyz, radius=self.voxel_r,
                                     nsample=self.voxel_sample)
        # transform rotation-variant coords into rotation-invariant coords
        new_points = new_points - pts_xyz.unsqueeze(2).repeat([1, 1, self.voxel_sample, 1])
        new_points = var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        new_points = new_points.permute(0, 3, 1, 2)  # (B, C_in, npoint, nsample), input features
        C_in = new_points.size()[1]
        nsample = new_points.size()[3]


        x = self.activation(self.bn_xyz_raising(self.xyz_raising(new_points)))
        x = F.max_pool2d(x, kernel_size=(1, nsample)).squeeze(3)  # (B, C_in, npoint)
        del new_points
        del pts_xyz
        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)

        x = self.conv_net(x)
        x = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))

        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x


class EnhancedFeaturePropagation(nn.Module):
    def __init__(self, k_dim, hidden_dim=384):
        super(EnhancedFeaturePropagation, self).__init__()

        # 将SpinNet特征从k_dim维转换到hidden_dim
        self.feature_proj = nn.Linear(k_dim, hidden_dim)

        # 处理点坐标
        self.coord_proj = nn.Linear(3, hidden_dim // 2)

        # MLP处理合并后的特征
        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, p,N, patch_feature):
        """
        p: (B,N,3) - 原始点云坐标
        patch_feature: (B,k,1) - SpinNet提取的特征
        """
        B, _, _ = p.shape

        # 1. 处理全局特征
        global_feat = patch_feature.squeeze(-1)  # (B,k)
        global_feat = self.feature_proj(global_feat)  # (B,hidden_dim)
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)  # (B,N,hidden_dim)

        # 2. 处理局部坐标信息
        local_feat = self.coord_proj(p)  # (B,N,hidden_dim//2)

        # 3. 合并特征
        combined_feat = torch.cat([global_feat, local_feat], dim=-1)  # (B,N,hidden_dim+hidden_dim//2)

        # 4. MLP处理 (使用1D卷积)
        combined_feat = combined_feat.transpose(1, 2)  # (B,hidden_dim+hidden_dim//2,N)
        point_features = self.mlp(combined_feat)  # (B,hidden_dim,N)
        point_features = point_features.transpose(1, 2)  # (B,N,hidden_dim)

        return point_features