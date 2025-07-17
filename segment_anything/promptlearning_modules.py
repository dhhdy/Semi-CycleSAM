import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.distributions import Normal, kl_divergence, Independent

def kl_gaussian_loss(mu, log_sigma):
    std = (2 * log_sigma).exp()  # σ² = exp(2 * log σ)
    kl_loss = -0.5 * torch.mean(1 + std - mu.pow(2) - std.exp())
    return kl_loss

class UNet_(nn.Module): # downsampling 3 blocks  BN?GN?
    def __init__(self,**kwargs):
        super(UNet_, self).__init__()
        print('load UNet_VAE_dn3')
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(192, 24, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool3d(2, stride=2)
        self.mp2 = nn.MaxPool3d(4, stride=4)
        self.mu1 = nn.Conv3d(24, 24, kernel_size=1, stride=1, padding=0)
        self.logvar1 = nn.Conv3d(24, 24, kernel_size=1, stride=1, padding=0)
        self.mu2 = nn.Linear(24, 24)
        self.logvar2 = nn.Linear(24, 24)
        self.up11 = nn.Linear(24, 192)
        self.up22 = nn.Linear(192, 768)
        self.up1 = nn.ConvTranspose3d(24, 192, kernel_size=4, stride=4, padding=0)
        self.up2 = nn.ConvTranspose3d(192, 384, kernel_size=2, stride=2, padding=0)

    def forward(self, x0):
        x1 = F.relu(self.conv1(x0))
        x1_ = self.mp1(x1)
        x2 = F.relu(self.conv2(x1_))
        x2_ = self.mp2(x2)
        x2__ = x2_.flatten(start_dim=1)
        # mu_mask = self.mu1(x2_)
        # log_mask = self.logvar1(x2_)
        # mu_point = self.mu2(x2__)
        # log_point = self.logvar2(x2__)
        # q_mask = Independent(Normal(loc=mu_mask, scale=torch.exp(log_mask)), 1)  # todo lgr: mask+log * 0.5
        # z_mask = q_mask.rsample()
        # q_point = Independent(Normal(loc=mu_point, scale=torch.exp(log_point)), 1)  # todo lgr: mask+log * 0.5
        # z_point = q_point.rsample()
        y2 = F.relu(self.up1(x2_) + x1_)
        y1 = F.relu(self.up2(y2) + x0)
        y22 = F.relu(self.up11(x2__))
        y11 = F.relu(self.up22(y22)).view(-1, 2, 384)

        return y11, y1, None

class UNet_VAE_dn3(nn.Module): # downsampling 3 blocks  BN?GN?
    def __init__(self,**kwargs):
        super(UNet_VAE_dn3, self).__init__()
        print('load UNet_VAE_dn3')
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(192, 24, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool3d(2, stride=2)
        self.mp2 = nn.MaxPool3d(4, stride=4)
        self.mu1 = nn.Conv3d(24, 24, kernel_size=1, stride=1, padding=0)
        self.logvar1 = nn.Conv3d(24, 24, kernel_size=1, stride=1, padding=0)
        self.mu2 = nn.Linear(24, 24)
        self.logvar2 = nn.Linear(24, 24)
        self.up11 = nn.Linear(24, 192)
        self.up22 = nn.Linear(192, 768)
        self.up1 = nn.ConvTranspose3d(24, 192, kernel_size=4, stride=4, padding=0)
        self.up2 = nn.ConvTranspose3d(192, 384, kernel_size=2, stride=2, padding=0)

    def forward(self, x0):
        x1 = F.relu(self.conv1(x0))
        x1_ = self.mp1(x1)
        x2 = F.relu(self.conv2(x1_))
        x2_ = self.mp2(x2)
        x2__ = x2_.flatten(start_dim=1)
        mu_mask = self.mu1(x2_)
        log_mask = self.logvar1(x2_)
        mu_point = self.mu2(x2__)
        log_point = self.logvar2(x2__)
        q_mask = Independent(Normal(loc=mu_mask, scale=torch.exp(log_mask)), 1)  # todo lgr: mask+log * 0.5
        z_mask = q_mask.rsample()
        q_point = Independent(Normal(loc=mu_point, scale=torch.exp(log_point)), 1)  # todo lgr: mask+log * 0.5
        z_point = q_point.rsample()
        mask_kl_loss = kl_gaussian_loss(mu_mask, log_mask)
        point_kl_loss = kl_gaussian_loss(mu_point, log_point)
        y2 = F.relu(self.up1(z_mask) + x1_)
        y1 = F.relu(self.up2(y2) + x0)
        y22 = F.relu(self.up11(z_point))
        y11 = F.relu(self.up22(y22)).view(-1, 2, 384)
        loss_total = mask_kl_loss + point_kl_loss

        return y11, y1, None, loss_total  # point mask None kl_loss


class Conv3DExpert_L(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3DExpert_L, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.mu = nn.Linear(out_channels, out_channels)
        self.lgr = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x1 = F.relu(self.fc(x))
        mu, lgr = self.mu(x1), self.lgr(x1)
        z = Independent(Normal(loc=mu, scale=torch.exp(lgr)), 1)
        return mu, lgr, z.rsample()


class Conv3DExpert_C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv3DExpert_C, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.mu = nn.Conv3d(out_channels, out_channels,
                              kernel_size=kernel_size-2,
                              padding=padding-1)
        self.lgr = nn.Conv3d(out_channels, out_channels,
                              kernel_size=kernel_size-2,
                              padding=padding-1)

    def forward(self, x):
        x1 = F.relu(self.conv(x))
        mu, lgr = self.mu(x1), self.lgr(x1)
        z = Independent(Normal(loc=mu, scale=torch.exp(lgr)), 1)
        return mu, lgr, z.rsample()

class mul_exp_UNet_VAE_dn3(nn.Module): # downsampling 3 blocks  BN?GN?
    def __init__(self,**kwargs):
        super(mul_exp_UNet_VAE_dn3, self).__init__()
        print('load mul_exp_UNet_VAE_dn3')
        self.num_experts = 2
        # 4 experts for 2 tasks
        self.experts_l = nn.ModuleList([Conv3DExpert_L(192, 24) for _ in range(self.num_experts)])
        self.experts_c = nn.ModuleList([Conv3DExpert_C(192, 24) for _ in range(self.num_experts)])
        self.gate_l = nn.Linear(192, self.num_experts)
        self.gate_c = nn.Conv3d(192, 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool3d(8, stride=8)
        self.up2 = nn.Linear(24, 768)
        self.up1 = nn.ConvTranspose3d(24, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, x0):
        x1 = F.relu(self.conv1(x0))
        x2 = self.mp(x1)
        x3 = x2.flatten(start_dim=1)

        expert_outputs_c = [expert(x1) for expert in self.experts_c]  # 每个元素是 (mu, lgr, z) 元组
        mu_mask = torch.stack([out[0] for out in expert_outputs_c], dim=1)
        lgr_mask = torch.stack([out[1] for out in expert_outputs_c], dim=1)
        z_mask = torch.stack([out[2] for out in expert_outputs_c], dim=1)

        expert_outputs_l = [expert(x3) for expert in self.experts_l]  # 每个元素是 (mu, lgr, z) 元组
        mu_point = torch.stack([out[0] for out in expert_outputs_l], dim=1)
        lgr_point = torch.stack([out[1] for out in expert_outputs_l], dim=1)
        z_point = torch.stack([out[2] for out in expert_outputs_l], dim=1)

        gate_scores_c = self.gate_c(x1)
        gate_scores_c = F.softmax(gate_scores_c, dim=1)
        expanded_gate_c = gate_scores_c.unsqueeze(2)
        z_mask = (expanded_gate_c * z_mask).sum(dim=1)
        gate_score_l = F.softmax(self.gate_l(x3), dim=-1)
        z_point = torch.bmm(gate_score_l.unsqueeze(1), z_point).squeeze(1)
        # add 24 ->192
        mask = F.relu(self.up1(z_mask))
        point = F.relu(self.up2(z_point)).view(-1, 2, 384)

        mask_kl_loss = sum(kl_gaussian_loss(mu_mask[:, i], lgr_mask[:, i]) for i in range(self.num_experts))
        point_kl_loss = sum(kl_gaussian_loss(mu_point[:, i], lgr_point[:, i]) for i in range(self.num_experts))
        loss_total = mask_kl_loss + point_kl_loss

        return point, mask, None, loss_total  # point mask None kl_loss

class mul_exp_UNet_VAE_dn3_PARA(nn.Module): # downsampling 3 blocks  BN?GN?
    def __init__(self,**kwargs):
        super(mul_exp_UNet_VAE_dn3_PARA, self).__init__()
        print('load mul_exp_UNet_VAE_dn3_PARA')
        self.num_experts = 2
        # 4 experts for 2 tasks
        self.experts_l = nn.ModuleList([Conv3DExpert_L(192, 24) for _ in range(self.num_experts)])
        self.experts_c = nn.ModuleList([Conv3DExpert_C(192, 24) for _ in range(self.num_experts)])
        self.gate_l = nn.Linear(192, self.num_experts)
        self.gate_c = nn.Conv3d(192, 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool3d(8, stride=8)
        self.up2 = nn.Linear(24, 768)
        self.skip_linear = nn.Linear(192, 768)
        self.skip_conv = nn.Conv3d(192, 384, kernel_size=1, stride=1, padding=0)
        self.up1 = nn.ConvTranspose3d(24, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, x0):
        x1 = F.relu(self.conv1(x0))
        x2 = self.mp(x1)
        x3 = x2.flatten(start_dim=1)

        # skip-connection
        x3_c = F.relu(self.skip_conv(x1))
        x3_l = F.relu(self.skip_linear(x3))

        expert_outputs_c = [expert(x1) for expert in self.experts_c]  # 每个元素是 (mu, lgr, z) 元组
        mu_mask = torch.stack([out[0] for out in expert_outputs_c], dim=1)
        lgr_mask = torch.stack([out[1] for out in expert_outputs_c], dim=1)
        z_mask = torch.stack([out[2] for out in expert_outputs_c], dim=1)

        expert_outputs_l = [expert(x3) for expert in self.experts_l]  # 每个元素是 (mu, lgr, z) 元组
        mu_point = torch.stack([out[0] for out in expert_outputs_l], dim=1)
        lgr_point = torch.stack([out[1] for out in expert_outputs_l], dim=1)
        z_point = torch.stack([out[2] for out in expert_outputs_l], dim=1)

        gate_scores_c = self.gate_c(x1)
        gate_scores_c = F.softmax(gate_scores_c, dim=1)
        expanded_gate_c = gate_scores_c.unsqueeze(2)
        z_mask = (expanded_gate_c * z_mask).sum(dim=1)
        gate_score_l = F.softmax(self.gate_l(x3), dim=-1)
        z_point = torch.bmm(gate_score_l.unsqueeze(1), z_point).squeeze(1)
        # add 24 ->192
        mask = F.relu(self.up1(z_mask) + x3_c)
        point = F.relu(self.up2(z_point) + x3_l).view(-1, 2, 384)

        mask_kl_loss = sum(kl_gaussian_loss(mu_mask[:, i], lgr_mask[:, i]) for i in range(self.num_experts))
        point_kl_loss = sum(kl_gaussian_loss(mu_point[:, i], lgr_point[:, i]) for i in range(self.num_experts))
        loss_total = mask_kl_loss + point_kl_loss

        return point, mask, None, loss_total  # point mask None kl_loss


class SimplePromptModule3D(nn.Module):
    def __init__(self,**kwargs):
        super(SimplePromptModule3D, self).__init__()
        self.num_center_channels = 768

        # 3D Convolutional layers for mask embedding
        self.conv1 = nn.Conv3d(384, self.num_center_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(self.num_center_channels, self.num_center_channels // 2, kernel_size=3, stride=1, padding=1)

        # Layers for points embedding
        self.conv3 = nn.Conv3d(self.num_center_channels, self.num_center_channels // 2, kernel_size=1, stride=1,
                               padding=0)

        maxpool_kernel = 8
        self.max_pool = nn.MaxPool3d(maxpool_kernel, stride=maxpool_kernel)

        pooled_volume = (8 // maxpool_kernel) ** 3
        self.fc = nn.Linear(self.num_center_channels // 2 * pooled_volume, 384 * 2)

    def forward(self, x):
        # Input shape: [N,384,8,8,8]

        # Mask embedding path
        out = F.relu(self.conv1(x))  # [B, num_center_channels, D, H, W]
        mask_embed = F.relu(self.conv2(out))  # [B, 384, D, H, W]

        # Points embedding path
        out = F.relu(self.conv3(out))  # [B, num_center_channels//2, D, H, W]
        out = self.max_pool(out)  # [B, num_center_channels//2, D//8, H//8, W//8]

        # Flatten and FC
        flat_out = out.flatten(start_dim=1)  # [B, num_center_channels//2 * 8*8*8]
        _points_embed = self.fc(flat_out)  # [B, 384*(num_points+1)]

        # Reshape to [B, 1, num_points+1, 256]
        points_embed = _points_embed.view(-1, 2, 384)

        # two_point:[N,2,384]spare  #one_mask[N,384,8,8,8] dense
        return points_embed, mask_embed, None

class SimplePromptModule3D_latent(nn.Module):
    def __init__(self,**kwargs):
        super(SimplePromptModule3D_latent, self).__init__()
        self.num_center_channels = 768

        # 3D Convolutional layers for mask embedding
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv3d(self.num_center_channels, self.num_center_channels // 2, kernel_size=3, stride=1, padding=1)

        # Layers for points embedding
        self.conv3 = nn.Conv3d(192, 384, kernel_size=1, stride=1,
                               padding=0)

        self.mu_s = nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1)
        self.logvar_s = nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1)

        self.mu_p = nn.Sequential(nn.Linear(384, 384), nn.ReLU(),
                                nn.Linear(384, 768))
        self.logvar_p = nn.Sequential(nn.Linear(384, 384), nn.ReLU(),
                                nn.Linear(384, 768))

        maxpool_kernel = 8
        self.max_pool = nn.MaxPool3d(maxpool_kernel, stride=maxpool_kernel)

        pooled_volume = (8 // maxpool_kernel) ** 3

    def forward(self, x, predict_mask=None):
        # Input shape: [N,384,8,8,8]

        # Mask embedding path
        out = F.relu(self.conv1(x))  # [B, num_center_channels, D, H, W]
        # mask_embed = F.relu(self.conv2(out))  # [B, 384, D, H, W]
        mask_mu = self.mu_s(out)
        mask_log = self.logvar_s(out)
        # Points embedding path
        out = F.relu(self.conv3(out))
        out = self.max_pool(out)

        # Flatten and FC
        out = out.flatten(start_dim=1)
        point_mu = self.mu_p(out)
        point_log = self.logvar_p(out)

        mask_q = Independent(Normal(loc=mask_mu, scale=torch.exp(mask_log)), 1)  # todo lgr: mask+log * 0.5
        mask_z = mask_q.rsample()
        point_q = Independent(Normal(loc=point_mu, scale=torch.exp(point_log)), 1)
        point_z = point_q.rsample()
        kl_loss = (kl_gaussian_loss(mask_mu, mask_log) + kl_gaussian_loss(point_mu, point_log))


        # Reshape to [B, 1, num_points+1, 256]
        points_embed = point_z.view(-1, 2, 384)
        mask_embed = mask_z
        # two_point:[N,2,384]spare  #one_mask[N,384,8,8,8] dense

        return points_embed, mask_embed, None, kl_loss

class SimplePromptModule3D_latent_dn(nn.Module):
    def __init__(self,**kwargs):
        super(SimplePromptModule3D_latent_dn, self).__init__()
        self.num_center_channels = 768

        # 3D Convolutional layers for mask embedding
        self.conv1 = nn.Conv3d(384, 192, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv3d(self.num_center_channels, self.num_center_channels // 2, kernel_size=3, stride=1, padding=1)

        # Layers for points embedding
        self.conv3 = nn.Conv3d(192, 384, kernel_size=1, stride=1,
                               padding=0)

        self.mu_s = nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1)
        self.logvar_s = nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1)

        self.mu_p = nn.Sequential(nn.Linear(384, 384), nn.ReLU(),
                                nn.Linear(384, 768))
        self.logvar_p = nn.Sequential(nn.Linear(384, 384), nn.ReLU(),
                                nn.Linear(384, 768))

        maxpool_kernel = 8
        self.max_pool = nn.MaxPool3d(maxpool_kernel, stride=maxpool_kernel)

        pooled_volume = (8 // maxpool_kernel) ** 3

    def forward(self, x, predict_mask=None):
        # Input shape: [N,384,8,8,8]

        # Mask embedding path
        out = F.relu(self.conv1(x))  # [B, num_center_channels, D, H, W]
        # mask_embed = F.relu(self.conv2(out))  # [B, 384, D, H, W]
        mask_mu = self.mu_s(out)
        mask_log = self.logvar_s(out)
        # Points embedding path
        out = F.relu(self.conv3(out))
        out = self.max_pool(out)

        # Flatten and FC
        out = out.flatten(start_dim=1)
        point_mu = self.mu_p(out)
        point_log = self.logvar_p(out)

        mask_q = Independent(Normal(loc=mask_mu, scale=torch.exp(mask_log)), 1)  # todo lgr: mask+log * 0.5
        mask_z = mask_q.rsample()
        point_q = Independent(Normal(loc=point_mu, scale=torch.exp(point_log)), 1)
        point_z = point_q.rsample()
        kl_loss = (kl_gaussian_loss(mask_mu, mask_log) + kl_gaussian_loss(point_mu, point_log))

        # Reshape to [B, 1, num_points+1, 256]
        points_embed = point_z.view(-1, 2, 384)
        mask_embed = mask_z
        # two_point:[N,2,384]spare  #one_mask[N,384,8,8,8] dense

        return points_embed, mask_embed, None, kl_loss

promptmodule_zoo = {
            "simple_module3D": SimplePromptModule3D,
            "simple_module3D_latent": SimplePromptModule3D_latent,  # 0.84
            "UNet_VAE_dn3": UNet_VAE_dn3,
            "UNet_": UNet_,
            "mul_exp_UNet_VAE_dn3": mul_exp_UNet_VAE_dn3,
            "mul_exp_UNet_VAE_dn3_PARA": mul_exp_UNet_VAE_dn3_PARA,
            }

# m = promptmodule_zoo["simple_module3D_latent"]().cuda()
# a = torch.zeros(1,384,8,8,8).cuda()
# b = m(a)
# print(b)
