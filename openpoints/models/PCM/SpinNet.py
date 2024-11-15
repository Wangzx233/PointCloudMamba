import os
import sys

from fontTools.unicodedata import script
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F

import ThreeDCCN as pn
import SpinNet.script.common as cm
from SpinNet.script.common import switch


class Descriptor_Net(nn.Module):
    def __init__(self, des_r, rad_n, azi_n, ele_n, voxel_r, voxel_sample, dataset):
        super(Descriptor_Net, self).__init__()
        # 半径
        self.des_r = des_r
        # 径向分区数
        self.rad_n = rad_n
        # 方位角分区数
        self.azi_n = azi_n
        # 仰角分区数
        self.ele_n = ele_n
        # 体素半径
        self.voxel_r = voxel_r
        # 体素采样数
        self.voxel_sample = voxel_sample
        # 数据集名称
        self.dataset = dataset

        # 批量归一化层
        self.bn_xyz_raising = nn.BatchNorm2d(16)
        self.bn_mapping = nn.BatchNorm2d(16)
        # 激活函数
        self.activation = nn.ReLU()
        # 一维卷积层，用于提升维度
        self.xyz_raising = nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        # 圆柱网络（Cylindrical_Net），来源于 L2Net 论文（CVPR17），一系列卷积操作提取图像的局部特征
        self.conv_net = pn.Cylindrical_Net(inchan=16, dim=32)

    def forward(self, input):
        #首先计算中心点 center，然后计算每个点相对于中心点的坐标 delta_x。
        center = input[:, -1, :].unsqueeze(1)
        delta_x = input[:, :, 0:3] - center[:, :, 0:3]  # (B, npoint, 3), normalized coordinates

        for case in switch(self.dataset):
            if case('3DMatch'):
                # 计算参考轴 z_axis 并进行旋转以消除旋转变化
                z_axis = cm.cal_Z_axis(delta_x, ref_point=input[:, -1, :3])
                z_axis = cm.l2_norm(z_axis, axis=1)
                R = cm.RodsRotatFormula(z_axis, torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(z_axis.shape[0], 1))
                delta_x = torch.matmul(delta_x, R)
                break
            if case('KITTI'):
                break

        # partition the local surface along elevator, azimuth, radial dimensions
        # 将局部表面划分为体素
        S2_xyz = torch.FloatTensor(cm.get_voxel_coordinate(radius=self.des_r,
                                                           rad_n=self.rad_n,
                                                           azi_n=self.azi_n,
                                                           ele_n=self.ele_n))

        # S2_xyz 包含所有体素中心球面坐标的张量，然后通过 view 和 repeat 操作将其转换为一个形状为 [batch_size, num_voxels, 3] 的张量，其中 batch_size 是批处理大小，num_voxels 是体素的数量。这样每个体素中心在笛卡尔坐标系中的坐标就被存储在 pts_xyz 中
        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()

        # query points in sphere
        new_points = cm.sphere_query(delta_x, pts_xyz, radius=self.voxel_r,
                                     nsample=self.voxel_sample)
        # transform rotation-variant coords into rotation-invariant coords
        new_points = new_points - pts_xyz.unsqueeze(2).repeat([1, 1, self.voxel_sample, 1])
        new_points = cm.var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        # 重新排列张量
        print(new_points.shape)
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C_in, npoint, nsample), input features
        print(new_points.shape)

        # C_in 输入的维度
        C_in = new_points.size()[1]

        # nsample 是领域内
        nsample = new_points.size()[3]

        # 升维，归一化，激活
        x = self.activation(self.bn_xyz_raising(self.xyz_raising(new_points)))

        # 最大池化，然后去掉最后一个维度
        x = F.max_pool2d(x, kernel_size=(1, nsample)).squeeze(3)  # (B, C_in, npoint)

        del new_points
        del pts_xyz
        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)

        # 圆柱网络的前向传播，用于进一步提取特征。
        x = self.conv_net(x)
        # 最大池化，用于降低特征图的空间维度。
        x = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))

        return x

    def get_parameter(self):
        return list(self.parameters())
