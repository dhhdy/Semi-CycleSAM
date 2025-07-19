import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.distributions import Normal, kl_divergence, Independent

def kl_gaussian_loss(mu, log_sigma):
    std = (2 * log_sigma).exp()  # σ² = exp(2 * log σ)
    kl_loss = -0.5 * torch.mean(1 + std - mu.pow(2) - std.exp())
    return kl_loss

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


promptmodule_zoo = {
            "mul_exp_UNet_VAE_dn3": mul_exp_UNet_VAE_dn3,
            }

