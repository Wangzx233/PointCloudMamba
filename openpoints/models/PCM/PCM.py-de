from ..build import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from .mamba_layer import MambaBlock
from .PCM_utils import MLP, serialization, _init_weights, index_points
from .PointMLP_layers import ConvBNReLU1D, LocalGrouper, PreExtraction, PreExtraction_Replace,\
    PosExtraction, get_activation, PointNetFeaturePropagation
from typing import List
from ..layers import furthest_point_sample
from .SpinNet import *


def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


def var_to_invar(pts, rad_n, azi_n, ele_n):
    """
    直接从原始SpinNet代码复制
    """
    device = pts.device
    B, N, nsample, C = pts.shape
    assert N == rad_n * azi_n * ele_n
    angle_step = np.array([0, 0, 2 * np.pi / azi_n])
    pts = pts.view(B, rad_n, ele_n, azi_n, nsample, C)

    R = np.zeros([azi_n, 3, 3])
    for i in range(azi_n):
        angle = -1 * i * angle_step
        r = angles2rotation_matrix(angle)
        R[i] = r
    R = torch.FloatTensor(R).to(device)
    R = R.view(1, 1, 1, azi_n, 3, 3).repeat(B, rad_n, ele_n, 1, 1, 1)
    new_pts = torch.matmul(pts, R.transpose(-1, -2))

    return new_pts.view(B, -1, nsample, C)


class Cylindrical_Net(nn.Module):
    """
    直接从原始SpinNet代码复制
    """

    def __init__(self, inchan=16, dim=32):
        super().__init__()
        self.outdim = dim

        # 保持原始SpinNet的网络结构
        self.ops = nn.ModuleList([
            nn.Conv3d(inchan, 32, k=[3, 3, 3], padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, k=[3, 3, 3], padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, k=[3, 3, 3], padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, k=[3, 3, 3], padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, dim, kernel_size=2, stride=2)
        ])

    def forward(self, x):
        """
        保持原始SpinNet的前向传播逻辑
        x: [B, C, rad_n, ele_n, azi_n]
        """
        for op in self.ops:
            if isinstance(op, nn.Conv3d):
                x = op(x)
            else:
                # 当需要从3D转为2D时
                if len(x.shape) == 5 and isinstance(op, nn.Conv2d):
                    x = x.squeeze(2)
                x = op(x)
        return x


class Descriptor_Net(nn.Module):
    """
    原始SpinNet的描述子网络
    """

    def __init__(self, config):
        super().__init__()
        self.des_r = config['des_r']
        self.rad_n = config['rad_n']
        self.azi_n = config['azi_n']
        self.ele_n = config['ele_n']
        self.voxel_r = config['voxel_r']
        self.voxel_sample = config['voxel_sample']

        self.bn_xyz_raising = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.xyz_raising = nn.Conv2d(3, 16, kernel_size=1)
        self.conv_net = Cylindrical_Net(inchan=16, dim=32)

    def forward(self, input):
        center = input[:, -1, :].unsqueeze(1)
        delta_x = input[:, :, 0:3] - center[:, :, 0:3]

        # 计算z轴和旋转矩阵
        z_axis = cal_Z_axis(delta_x, ref_point=input[:, -1, :3])
        z_axis = l2_norm(z_axis, axis=1)
        R = RodsRotatFormula(z_axis,
                             torch.FloatTensor([0, 0, 1]).to(z_axis.device).unsqueeze(0).repeat(z_axis.shape[0], 1))
        delta_x = torch.matmul(delta_x, R)

        # 获取球形网格坐标
        S2_xyz = get_voxel_coordinate(radius=self.des_r, rad_n=self.rad_n,
                                      azi_n=self.azi_n, ele_n=self.ele_n).to(delta_x.device)
        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1])

        # 球形查询
        new_points = sphere_query(delta_x, pts_xyz, radius=self.voxel_r, nsample=self.voxel_sample)
        new_points = new_points - pts_xyz.unsqueeze(2).repeat([1, 1, self.voxel_sample, 1])
        new_points = var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        x = new_points.permute(0, 3, 1, 2)
        x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        x = F.max_pool2d(x, kernel_size=(1, x.size(3))).squeeze(3)

        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)
        x = self.conv_net(x)
        x = F.adaptive_max_pool2d(x, 1)

        return x


class SpinNetFeatureExtraction(nn.Module):
    """
    SpinNet特征提取器与PCM的接口层
    """

    def __init__(self, in_channels, out_channels, spinnet_config, bias=True):
        super().__init__()
        self.spinnet = Descriptor_Net(spinnet_config)
        self.proj = nn.Linear(self.spinnet.conv_net.outdim, out_channels, bias=bias)

    def forward(self, xyz, points):
        B, N, _ = xyz.shape

        # FPS采样关键点
        fps_idx = furthest_point_sample(xyz, N // 4).long()
        kpts = index_points(xyz, fps_idx)

        # 准备SpinNet输入
        input_points = torch.cat([xyz, kpts], dim=1)  # 包含所有点和关键点

        # 提取SpinNet特征
        features = self.spinnet(input_points)  # [B, C, 1, 1]
        features = features.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, N//4, C]

        # 特征插值回原始分辨率
        features = feature_interpolate(features, kpts, xyz)

        # 投影到所需维度
        features = self.proj(features)

        return xyz, features.transpose(1, 2)

@MODELS.register_module()
class PointMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
                 spinnet_config=None, mamba_blocks=[1, 1, 1, 1],
                 mamba_layers_orders='z', cls_pooling="max",
                 use_order_prompt=False, prompt_num_per_order=1,
                 rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
                 bimamba_type="none", drop_path_rate=0.1, **kwargs):
        super(PointMambaEncoder, self).__init__()

        # 基本参数初始化
        self.stages = len(pre_blocks)
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.spinnet_blocks = nn.ModuleList()
        self.mamba_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        self.residual_proj_blocks_list = nn.ModuleList()
        self.cls_pooling = cls_pooling

        # 处理mamba相关参数
        self.mamba_layers_orders = mamba_layers_orders if isinstance(mamba_layers_orders, list) else [
                                                                                                         mamba_layers_orders] * sum(
            mamba_blocks)
        self.order = 'original'

        # 初始化处理层
        last_channel = embed_dim
        norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5)

        # 构建网络各阶段
        mamba_layer_idx = 0
        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]

            # SpinNet特征提取
            spinnet_block = SpinNetFeatureExtraction(
                last_channel,
                out_channel,
                spinnet_config,
                bias=bias
            )
            self.spinnet_blocks.append(spinnet_block)

            # 残差投影
            if last_channel == out_channel or i == 0:
                self.residual_proj_blocks_list.append(nn.Identity())
            else:
                self.residual_proj_blocks_list.append(nn.Linear(last_channel, out_channel, bias=False))

            # Mamba块
            mamba_block = nn.Sequential()
            for n_mamba in range(mamba_blocks[i]):
                mamba_block_module = MambaBlock(
                    dim=out_channel,
                    layer_idx=mamba_layer_idx,
                    bimamba_type=bimamba_type,
                    norm_cls=norm_cls,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                    drop_path=drop_path_rate
                )
                mamba_block.append(mamba_block_module)
                mamba_layer_idx += 1
            self.mamba_blocks_list.append(mamba_block)

            # 位置编码块
            self.pos_blocks_list.append(nn.Identity())

            last_channel = out_channel

        self.out_channels = last_channel
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward_cls_feat(self, p, x=None):
        self.order = "original"

        # 输入处理
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()

        batch_size, _, _ = x.size()
        x = self.embedding(x.transpose(1, 2)).transpose(1, 2)

        x_res = None
        mamba_layer_idx = 0

        # 前向传播
        for i in range(self.stages):
            # SpinNet特征提取
            p, x = self.spinnet_blocks[i](p, x)

            # Mamba处理
            x = x.permute(0, 2, 1).contiguous()
            x_res = self.residual_proj_blocks_list[i](x_res) if x_res is not None else None

            for layer in self.mamba_blocks_list[i]:
                p, x, x_res = self.serialization_func(
                    p, x, x_res,
                    self.mamba_layers_orders[mamba_layer_idx]
                )
                x, x_res = layer(x, x_res)
                mamba_layer_idx += 1

            x = x.permute(0, 2, 1).contiguous()
            x = self.pos_blocks_list[i](x)

        # 全局池化
        if self.cls_pooling == "max":
            x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x = x.mean(dim=-1)
        else:  # mean + max
            x_max = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
            x_mean = x.mean(dim=-1)
            x = x_max + x_mean

        return x

    def forward(self, x, cls_label=None):
        return self.forward_cls_feat(x)

    def serialization_func(self, p, x, x_res, order):
        if order == self.order:
            return p, x, x_res
        else:
            p, x, x_res = serialization(p, x, x_res=x_res, order=order)
            self.order = order
            return p, x, x_res

    def index_points(points, idx):
        """
        Input:
            points: [B, N, C]
            idx: [B, S]
        Return:
            new_points: [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def feature_interpolate(features, kpts, query_pts):
        """
        Input:
            features: [B, K, C]
            kpts: [B, K, 3]
            query_pts: [B, N, 3]
        Return:
            interpolated_features: [B, N, C]
        """
        B, K, C = features.shape
        _, N, _ = query_pts.shape

        # 计算距离和权重
        dist = torch.cdist(query_pts, kpts)  # B,N,K
        weights = 1.0 / (dist + 1e-8)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # B,N,K

        # 特征插值
        interpolated_features = torch.matmul(weights, features)  # B,N,C
        return interpolated_features

    def forward_seg_feat(self, p, x=None):
        self.order = "original"
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x = torch.cat([x, p.transpose(1, 2)], dim=1).contiguous()
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N

        p_list, x_list = [p], [x]

        x_res = None

        pos_proj_idx = 0
        mamba_layer_idx = 0
        for i in range(self.stages):
            # Give p[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            p, x, x_res = self.local_grouper_list[i](p, x.permute(0, 2, 1), x_res)  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]

            x = x.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res = None
            x_res = self.residual_proj_blocks_list[i](x_res)
            for layer in self.mamba_blocks_list[i]:
                p, x, x_res = self.serialization_func(p, x, x_res, self.mamba_layers_orders[mamba_layer_idx])
                if self.use_windows:
                    p, x, x_res, n_windows, p_base, p_std = self.pre_split_windows(
                        p, x, x_res, windows_size=self.windows_size[i])
                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x = x + self.pos_proj(p)
                        else:
                            x = x + self.pos_proj[i](p)
                    else:
                        x = x + self.pos_proj[pos_proj_idx](p)
                        pos_proj_idx += 1
                if self.use_order_prompt:
                    layer_order_prompt_indexes = self.per_layer_prompt_indexe[mamba_layer_idx]
                    layer_order_prompt = self.order_prompt.weight[
                                         layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]
                    layer_order_prompt = self.order_prompt_proj[i](layer_order_prompt)
                    layer_order_prompt = layer_order_prompt.unsqueeze(0).repeat(p.shape[0], 1, 1)
                    x = torch.cat([layer_order_prompt, x, layer_order_prompt], dim=1)
                    if x_res is not None:
                        x_res = torch.cat([layer_order_prompt, x_res, layer_order_prompt], dim=1)
                    x, x_res = layer(x, x_res)
                    x = x[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res = x_res[:, self.promot_num_per_order:-self.promot_num_per_order]
                else:
                    x, x_res = layer(x, x_res)
                if self.use_windows:
                    p, x, x_res = self.post_split_windows(p, x, x_res, n_windows, p_base, p_std)
                mamba_layer_idx += 1
            x = x.permute(0, 2, 1).contiguous()
            x = self.pos_blocks_list[i](x)  # [b,d,g]

            p_list.append(p)
            x_list.append(x)

        return p_list, x_list

    def pre_split_windows(self, p, x, x_res, windows_size=1024):
        # x (bs, n, c), p (bs, n, 3)
        bs, n, c = x.shape
        if n <= windows_size:
            return p, x, x_res, 1, 0, 1

        # fps sample
        n_sample = n // windows_size * windows_size
        fps_idx = furthest_point_sample(p, n_sample).long()  # [B, n_windows]
        fps_idx = torch.sort(fps_idx, dim=-1)[0]
        new_p = index_points(p, fps_idx)  # [B, n_windows, 3]
        new_x = index_points(x, fps_idx)
        if x_res is not None:
            new_x_res = index_points(x_res, fps_idx)
        else:
            new_x_res = None

        # split windows
        bs, n, c = new_x.shape
        n_splits = n // windows_size
        new_p = new_p.reshape(bs, n_splits, windows_size, -1).flatten(0, 1)
        new_x = new_x.reshape(bs, n_splits, windows_size, -1).flatten(0, 1)
        if new_x_res is not None:
            new_x_res = new_x_res.reshape(bs, n_splits, windows_size, -1).flatten(0, 1).contiguous()

        p_base = torch.min(new_p, dim=2, keepdim=True)[0]
        p_std = torch.max(new_p, dim=2, keepdim=True)[0] - p_base + 1e-6
        new_p = (new_p - p_base) / p_std
        return new_p.contiguous(), new_x.contiguous(), new_x_res, n_splits, p_base, p_std

    def post_split_windows(self, p, x, x_res, n_windos, p_base, p_std):
        p = p * p_std + p_base
        if n_windos == 1:
            return p, x, x_res
        bs_nw, window_size, c = x.shape
        bs = bs_nw // n_windos

        p = p.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
        x = x.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
        if x_res is not None:
            x_res = x_res.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
            return p.contiguous(), x.contiguous(), x_res.contiguous()
        else:
            return p.contiguous(), x.contiguous(), x_res

@MODELS.register_module()
class PointMambaPartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int] = [384, 384, 384, 768, 768],
                 decoder_channel_list: List[int] = [768, 384, 384, 384],
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 mamba_blocks: List[int] = [1, 1, 1, 1],
                 mamba_layers_orders=['xyz', 'xyz', 'xyz', 'null'],
                 act_args: str = 'relu',
                 rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
                 bimamba_type="v2",
                 gmp_dim=64,cls_dim=64, bias=True,
                 **kwargs
                 ):
        super().__init__()

        ### Building Decoder #####
        print(encoder_channel_list)
        en_dims = encoder_channel_list
        de_dims = decoder_channel_list
        de_blocks = decoder_blocks
        if mamba_blocks[-1] != 0:
            assert mamba_layers_orders[-1] == "null"
        self.mamba_layers_orders = mamba_layers_orders
        self.decode_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        self.order = 'original'
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5,
        )
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1 == len(mamba_blocks) + 1
        assert sum(mamba_blocks) == len(mamba_layers_orders) or sum(mamba_blocks) == 0

        mamba_layer_idx = 0
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i] + en_dims[i + 1], de_dims[i + 1],
                                           blocks=de_blocks[i], res_expansion=1.0,
                                           bias=True, activation=act_args)
            )

            out_channel = de_dims[i + 1]
            mamba_block = nn.Sequential()
            mamba_block_num = mamba_blocks[i]
            for n_mamba in range(mamba_block_num):
                mamba_block_module = MambaBlock(dim=out_channel, layer_idx=mamba_layer_idx, bimamba_type=bimamba_type,
                                                norm_cls=norm_cls, fused_add_norm=fused_add_norm,
                                                residual_in_fp32=residual_in_fp32, drop_path=0.0)
                mamba_block.append(mamba_block_module)
                mamba_layer_idx += 1
            self.mamba_list.append(mamba_block)

        self.act = get_activation(act_args)

        # class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=act_args),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=act_args)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=act_args))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=act_args)
        self.out_channels = out_channel + gmp_dim + cls_dim

    def serialize_func(self, p, x, x_res, order, layers_outputs=[]):
        if order == self.order or order == "null":
            return p, x, x_res
        else:
            p, x, x_res = serialization(p, x, x_res=x_res, order=order, layers_outputs=layers_outputs)
            self.order = order
            return p, x, x_res

    def forward(self, p, f, cls_label):
        self.order = "original"
        B, N = p[0].shape[0:2]
        # here is the decoder
        xyz_list = p
        x_list = f

        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]

        mamba_layer_idx = 0
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)
            p = xyz_list[i + 1]

            # perform mamba
            x = x.permute(0, 2, 1).contiguous()
            x_res = None
            if len(self.mamba_list[i]) != 0:
                for layer in self.mamba_list[i]:
                    p, x, x_res = self.serialize_func(p, x, x_res, self.mamba_layers_orders[mamba_layer_idx])
                    x, x_res = layer(x, x_res)
                    mamba_layer_idx += 1
            x = x.permute(0, 2, 1).contiguous()


        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # [b, gmp_dim, 1]

        # here is the cls_token
        cls_one_hot = torch.zeros((B, 16), device=p[0].device)
        cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)

        cls_token = self.cls_map(cls_one_hot)  # [b, cls_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        return x


@MODELS.register_module()
class PointMambaDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int] = [384, 384, 384, 768, 768],
                 decoder_channel_list: List[int] = [768, 384, 384, 384],
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 mamba_blocks: List[int] = [1, 1, 1, 1],
                 mamba_layers_orders=['xyz', 'xyz', 'xyz', 'null'],
                 act_args: str = 'relu',
                 rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
                 bimamba_type="v2",
                 gmp_dim=64, bias=True,
                 **kwargs
                 ):
        super().__init__()

        ### Building Decoder #####
        print(encoder_channel_list)
        en_dims = encoder_channel_list
        de_dims = decoder_channel_list
        de_blocks = decoder_blocks
        if mamba_blocks[-1] != 0:
            assert mamba_layers_orders[-1] == "null"
        self.mamba_layers_orders = mamba_layers_orders
        self.decode_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        self.order = 'original'
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5,
        )
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1 == len(mamba_blocks) + 1
        assert sum(mamba_blocks) == len(mamba_layers_orders) or sum(mamba_blocks) == 0

        mamba_layer_idx = 0
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i] + en_dims[i + 1], de_dims[i + 1],
                                           blocks=de_blocks[i], res_expansion=1.0,
                                           bias=True, activation=act_args)
            )

            out_channel = de_dims[i + 1]
            mamba_block = nn.Sequential()
            mamba_block_num = mamba_blocks[i]
            for n_mamba in range(mamba_block_num):
                mamba_block_module = MambaBlock(dim=out_channel, layer_idx=mamba_layer_idx, bimamba_type=bimamba_type,
                                                norm_cls=norm_cls, fused_add_norm=fused_add_norm,
                                                residual_in_fp32=residual_in_fp32, drop_path=0.0)
                mamba_block.append(mamba_block_module)
                mamba_layer_idx += 1
            self.mamba_list.append(mamba_block)

        self.act = get_activation(act_args)

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=act_args))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=act_args)
        self.out_channels = out_channel + gmp_dim

    def serialize_func(self, p, x, x_res, order, layers_outputs=[]):
        if order == self.order or order == "null":
            return p, x, x_res
        else:
            p, x, x_res = serialization(p, x, x_res=x_res, order=order, layers_outputs=layers_outputs)
            self.order = order
            return p, x, x_res

    def forward(self, p, f):
        self.order = "original"
        B, N = p[0].shape[0:2]
        # here is the decoder
        xyz_list = p
        x_list = f

        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]

        mamba_layer_idx = 0
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)
            p = xyz_list[i + 1]

            # perform mamba
            x = x.permute(0, 2, 1).contiguous()
            x_res = None
            if len(self.mamba_list[i]) != 0:
                for layer in self.mamba_list[i]:
                    p, x, x_res = self.serialize_func(p, x, x_res, self.mamba_layers_orders[mamba_layer_idx])
                    x, x_res = layer(x, x_res)
                    mamba_layer_idx += 1
            x = x.permute(0, 2, 1).contiguous()


        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # [b, gmp_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), ], dim=1)
        return x

