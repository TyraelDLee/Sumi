import math
import torch
import torch.nn as nn
from torch import einsum
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
import math
from utils import misc

from model.SymmCompletion6 import fps_subsample


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = pointnet2_utils.gather_operation(xyz, pointnet2_utils.furthest_point_sample(xyz_flipped,
                                                                                          npoint))  # (B, 3, npoint)
    if idx is None:
        # idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
        idx = query_knn_point(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = pointnet2_utils.grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = pointnet2_utils.grouping_operation(points, idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def query_knn_point(k, xyz, new_xyz):
    """
    Input:
        x: point cloud, [B, N, 3]
    Return:
        group_xyz: [B, N, K]
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()

def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        # 点云的self-attention计算的是一个领域内点与点之间的关系

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()

        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)

        key = self.conv_key(x)

        value = self.conv_value(x)

        query = self.conv_query(x)

        key = pointnet2_utils.grouping_operation(key, idx_knn)  # b, dim, n, n_knn

        qk_rel = query.reshape((b, -1, n, 1)) - key

        # 获取点与
        pos_rel = pos.reshape((b, -1, n, 1)) - pointnet2_utils.grouping_operation(pos, idx_knn)  # b, 3, n, n_knn

        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = pointnet2_utils.grouping_operation(value, idx_knn) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.af = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.af(out)

        return out


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None, no_group=False):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if no_group:
            new_points = self.mlp_conv(xyz)
            new_points = torch.max(new_points, 3)[0]
            return new_points

        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample,
                                                                         self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points



class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        self.af = nn.ReLU()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(self.af)
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)



class CrossFormer(nn.Module):

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = pointnet2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(),
                                               pointnet2_utils.furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

class LSTNet(nn.Module):
    def __init__(self, out_dim=512, config=None):
        super(LSTNet, self).__init__()
        self.sa_module_gt = PointNet_SA_Module_KNN(1024, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_ca = PointNet_SA_Module_KNN(1024, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.transformer_gt = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])

        self.mlp = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 9 + 3)
        )

        self.mask_token = nn.Conv1d(3, 512, 1)

        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        self.time_step = config.num_steps
        self.betas = self.linear_schedule(timesteps=self.time_step)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, axis=0)
        self.alpha_bar_t_minus_one = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sigma = self.betas * (1.0 - self.alpha_bar_t_minus_one) / (1.0 - self.alpha_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bar_minus_one = torch.sqrt(self.alpha_bar_t_minus_one)

        self.time_emb = nn.Sequential(
            TimeEmbedding(384),
            nn.Linear(384, 384),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3)
        )

    def linear_schedule(self, timesteps):
        return torch.linspace(self.beta_1, self.beta_T, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        b = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0).to(x_0.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_bar, t, x_0.shape).to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(
            x_0.device)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, point_cloud, gt=None, t=None):
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        # get the key points and its features
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        keyfeatures = self.transformer_1(keyfeatures, keypoints)  # B,128,512

        feat = self.expanding(keyfeatures)
        feat = feat.transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1)  # B,640,512

        ret = self.mlp(feat)
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)  # B, 1024, 3
        gt = fps_subsample(gt.transpose(2, 1).contiguous(), 1024).transpose(2, 1).contiguous()
        x_t, _ = self.forward_diffusion(gt, t)
        # print(x_t.shape)
        keypoints_noise, keyfeatures_noise, _ = self.sa_module_gt(x_t, x_t)
        _, coarse_feature, _ = self.sa_module_ca(coarse, coarse)
        keypoints_noise = torch.cat([keypoints_noise, coarse], dim=-1)
        keyfeatures_noise = torch.cat([keyfeatures_noise, coarse_feature], dim=-1)
        keyfeatures_noise = self.transformer_gt(keyfeatures_noise, keypoints_noise).transpose(1,2)
        keypoints_noise = self.fc(keyfeatures_noise)[:, :1024, :]

        return keypoints_noise, symmetry_points, keyfeatures

    def sampling_t(self, point_cloud, x_t, i):
        b, _, n = point_cloud.shape
        t = torch.full((1,), i, device=point_cloud.device)
        # q_t = self.time_emb(t.to(coarse.device)).unsqueeze(1).expand(-1, 1024, -1)
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape).to(point_cloud.device)

        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        # get the key points and its features
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        keyfeatures = self.transformer_1(keyfeatures, keypoints)  # B,128,512

        feat = self.expanding(keyfeatures)
        feat = feat.transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1)  # B,640,512

        ret = self.mlp(feat)
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)  # B, 1024, 3
        keypoints_noise, keyfeatures_noise, _ = self.sa_module_gt(x_t, x_t)
        _, coarse_feature, _ = self.sa_module_ca(coarse, coarse)
        keypoints_noise = torch.cat([keypoints_noise, coarse], dim=-1)
        keyfeatures_noise = torch.cat([keyfeatures_noise, coarse_feature], dim=-1)
        keyfeatures_noise = self.transformer_gt(keyfeatures_noise, keypoints_noise).transpose(1, 2)
        x_rec = self.fc(keyfeatures_noise)[:, :1024, :].transpose(2, 1).contiguous()

        alpha_bar_t = self.get_index_from_list(self.alpha_bar, t, x_t.shape).to(coarse.device)
        alpha_bar_t_minus_one = self.get_index_from_list(self.alpha_bar_t_minus_one, t, x_t.shape).to(coarse.device)
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas, t, x_t.shape).to(coarse.device)
        sqrt_alphas_bar_t_minus_one = self.get_index_from_list(self.sqrt_alpha_bar_minus_one, t, x_t.shape).to(
            coarse.device)
        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one)) / (1 - alpha_bar_t) * x_t + (
                sqrt_alphas_bar_t_minus_one * betas_t) / (1 - alpha_bar_t) * x_rec

        sigma_t = self.get_index_from_list(self.sigma, t, x_t.shape).to(coarse.device)

        if t == 0:
            return model_mean, symmetry_points, keyfeatures
        else:
            return model_mean + torch.sqrt(sigma_t) * x_rec, symmetry_points, keyfeatures

    def sampling(self, point_cloud, noise_patch=None):
        B, _, N = point_cloud.shape

        if noise_patch is None:
            noise_patch = torch.randn((B, 3, 1024)).to(point_cloud.device)
        traj = []

        for i in range(0, self.time_step)[::-1]:
            noise_patch, symmetry_points, keyfeatures = self.sampling_t(point_cloud, noise_patch, i)

        return noise_patch, symmetry_points, keyfeatures


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()

        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)


    def forward(self, feat_x, feat_y):
        # cross attention
        feat = self.corssformer_1(feat_x, feat_y)

        # self attention
        feat = self.corssformer_2(feat, feat)
        return feat


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for the diffusion process.

    Input:
    Timestep: the current timestep, in range [1, ..., T]
    """

    def __init__(self, dim):
        super().__init__()
        self.emb_dim = dim

    def forward(self, ts):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=ts.device) * -emb)
        emb = ts[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SGFormer(nn.Module):
    def __init__(self, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat)
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)  # B, N * up_ratio, 3
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1,
                                                                                                           3) + offset
        return pcd_up


class Difformer(nn.Module):
    def __init__(self, up_factor, config):
        super(Difformer, self).__init__()
        self.up_factor = up_factor
        # self.local_mlp = LocalPointMLP()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_ds = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.mlp_1_ds = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2_ds = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer_ds = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_d = Fusion(in_channel=256)
        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

        self.mask_token = nn.Conv1d(3, 512, 1)

        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        self.time_step = config.num_steps
        self.betas = self.linear_schedule(timesteps=self.time_step)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, axis=0)
        self.alpha_bar_t_minus_one = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sigma = self.betas * (1.0 - self.alpha_bar_t_minus_one) / (1.0 - self.alpha_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bar_minus_one = torch.sqrt(self.alpha_bar_t_minus_one)

        self.time_emb = nn.Sequential(
            TimeEmbedding(384),
            nn.Linear(384, 384),
            nn.ReLU()
        )

    def linear_schedule(self, timesteps):
        return torch.linspace(self.beta_1, self.beta_T, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        b = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0).to(x_0.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_bar, t, x_0.shape).to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(
            x_0.device)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, coarse, symmetry_feat, partial_feat, gt=None, t=None):
        b, _, n = coarse.shape

        x_t, _ = self.forward_diffusion(gt, t)

        idx = query_knn_point(16, x_t.contiguous(), coarse.transpose(1, 2).contiguous())

        ds = misc.gather_points(x_t, idx.int()).transpose(1, 2).contiguous()

        feat_ds = self.mlp_1_ds(ds)
        feat_max_ds = feat_ds.max(dim=-1, keepdim=True)[0]
        feat_ds = torch.cat([feat_ds, feat_max_ds.repeat(1, 1, feat_ds.shape[-1])], dim=1)
        feat_ds = self.mlp_2_ds(feat_ds)
        feat_ds = self.transformer_ds(feat_ds, coarse)
        feat_ds = self.expand_dim_ds(feat_ds).transpose(2, 1).contiguous()

        # feat = self.mlp_1(coarse)
        # feat_max = feat.max(dim=-1, keepdim=True)[0]
        # feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        # feat = self.mlp_2(feat)
        # feat = self.transformer(feat, coarse)

        # feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        # feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # noise awareness
        # feat = self.fusion_d(feat, feat_ds)
        # partial part awareness
        feat_p = self.fusion_1(feat_ds, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat_ds, symmetry_feat)
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)  # B, N * up_ratio, 3
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1,
                                                                                                           3) + offset

        return pcd_up

    def sampling_t(self, coarse, symmetry_feat, partial_feat, x_t, i):
        b, _, n = coarse.shape
        t = torch.full((1,), i, device=coarse.device)
        # q_t = self.time_emb(t.to(coarse.device)).unsqueeze(1).expand(-1, 1024, -1)
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape).to(coarse.device)

        idx = query_knn_point(16, x_t.contiguous(), coarse.transpose(1, 2).contiguous())

        ds = misc.gather_points(x_t, idx.int()).transpose(1, 2).contiguous()

        feat_ds = self.mlp_1_ds(ds)
        feat_max_ds = feat_ds.max(dim=-1, keepdim=True)[0]
        feat_ds = torch.cat([feat_ds, feat_max_ds.repeat(1, 1, feat_ds.shape[-1])], dim=1)
        feat_ds = self.mlp_2_ds(feat_ds)
        feat_ds = self.transformer_ds(feat_ds, coarse)
        feat_ds = self.expand_dim_ds(feat_ds).transpose(2, 1).contiguous()

        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # noise awareness
        feat = self.fusion_d(feat, feat_ds)
        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat)
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)  # B, N * up_ratio, 3
        x_rec = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1,
                                                                                                          3) + offset

        alpha_bar_t = self.get_index_from_list(self.alpha_bar, t, x_t.shape).to(coarse.device)
        alpha_bar_t_minus_one = self.get_index_from_list(self.alpha_bar_t_minus_one, t, x_t.shape).to(coarse.device)
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas, t, x_t.shape).to(coarse.device)
        sqrt_alphas_bar_t_minus_one = self.get_index_from_list(self.sqrt_alpha_bar_minus_one, t, x_t.shape).to(
            coarse.device)
        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one)) / (1 - alpha_bar_t) * x_t + (
                sqrt_alphas_bar_t_minus_one * betas_t) / (1 - alpha_bar_t) * x_rec

        sigma_t = self.get_index_from_list(self.sigma, t, x_t.shape).to(coarse.device)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(sigma_t) * x_rec

    def sampling(self, coarse, symmetry_feat, partial_feat, noise_patch=None):
        B, _, N = coarse.shape

        if noise_patch is None:
            noise_patch = torch.randn((B, N * self.up_factor, 3)).to(coarse.device)
        traj = []

        for i in range(0, self.time_step)[::-1]:
            noise_patch = self.sampling_t(coarse, symmetry_feat, partial_feat, noise_patch, i)

        return noise_patch.reshape(B, -1, 3)


class local_encoder(nn.Module):
    def __init__(self, out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)

        return feat


class SumiCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # self.up_factors = [int(i) for i in config.up_factors.split(',')]
        self.lstnet = LSTNet(out_dim=config.lst_dim, config=config)
        self.local_encoder = local_encoder(out_channel=config.encoder_dim)
        self.sumi_module = SGFormer(up_factor=config.up_factor[0])
        self.sgformer_2 = SGFormer(up_factor=config.up_factor[1])
        self.include_input = config.include_input
        if config.loss is 'cdl1':
            self.loss_func = ChamferDistanceL1()
            print("Using CDL1")
        elif config.loss is 'cdl2':
            self.loss_func = ChamferDistanceL2()
            print("Using CDL2")
        else:
            Exception('No imp')
        self.num_output = config.num_output

    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1], loss_list[0], loss_list[-1]

    def forward(self, point_cloud, gt=None, t=None):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        if self.training:
            assert gt is not None
            assert t is not None
            coarse, symmetry_points, keyfeatures = self.lstnet(point_cloud.transpose(2, 1).contiguous(),
                                                               gt.transpose(2, 1).contiguous(), t)
            # coarse size: 1024
            feat_symmetry = self.local_encoder(symmetry_points)  # B,128,512
            feat_partial = keyfeatures  # B,128,512
            fine1 = self.sumi_module(coarse.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)
            fine2 = self.sgformer_2(fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)
            #

            if self.include_input:
                fine2 = torch.cat([fine2, point_cloud],dim=1).contiguous()


            rets = [coarse.transpose(2, 1).contiguous(), fine1, fine2]
        else:
            assert gt is None
            assert t is None
            coarse, symmetry_points, keyfeatures = self.lstnet.sampling(point_cloud.transpose(2, 1).contiguous())
            # coarse size: 1024
            feat_symmetry = self.local_encoder(symmetry_points)  # B,128,512
            feat_partial = keyfeatures  # B,128,512
            fine1 = self.sumi_module(coarse, feat_symmetry, feat_partial)
            fine2 = self.sgformer_2(fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)

            if self.include_input:
                fine2 = torch.cat([fine2, point_cloud],dim=1).contiguous()

            rets = [coarse.transpose(2, 1).contiguous(), fine1, fine2]

        # print(fine2.shape)
        assert fine2.shape[1] == self.num_output
        self.pred_dense_point = rets[-1]

        return rets



# todo: keep the sgformer for 2k point then use diffuser for 16k
# todo: try use the 6.43 one diffuser here too.