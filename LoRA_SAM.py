import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import numpy as np
import random
import math
import torch.nn as nn
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.modeling import sam3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas
from torch.utils.data.dataloader import default_collate
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

# import torchio as tio
from skimage.measure import label

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry

from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL_Val
from segment_anything.promptlearning_modules import promptmodule_zoo
from safetensors import safe_open
from safetensors.torch import save_file
class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :, :self.dim] += new_q
        qkv[:, :, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam3D(nn.Module):
    def __init__(self, sam_model: sam3D, r: int, lora_layer=None):
        super(LoRA_Sam3D, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        # self.auto_prompt = promptmodule_zoo["mul_exp_UNet_VAE_dn3_PARA"]()
        self.auto_prompt = promptmodule_zoo["mul_exp_UNet_VAE_dn3"]()
        # self.auto_prompt = promptmodule_zoo["UNet_VAE_dn3"]()
        for param in sam_model.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        # Add LoRA to mask decoder
        self.mask_decoder_w_As = nn.ModuleList()
        self.mask_decoder_w_Bs = nn.ModuleList()

        for module in sam_model.mask_decoder.modules():
            if isinstance(module, nn.Linear):
                w_a = nn.Linear(module.in_features, r, bias=False)
                w_b = nn.Linear(r, module.out_features, bias=False)
                self.mask_decoder_w_As.append(w_a)
                self.mask_decoder_w_Bs.append(w_b)

                # Wrap the original linear layer
                new_module = _LoRA_qkv(module, w_a, w_b, w_a, w_b)

                # Replace the original module with the wrapped one
                for name, child in sam_model.mask_decoder.named_children():
                    if child is module:
                        setattr(sam_model.mask_decoder, name, new_module)
                        break

        self.reset_parameters()
        self.sam = sam_model

    @property
    def image_encoder(self):
        return self.sam.image_encoder

    @property
    def prompt_encoder(self):
        return self.sam.prompt_encoder

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)

    def reset_parameters(self) -> None:
        for w_A in self.w_As + self.mask_decoder_w_As:
            nn.init.trunc_normal_(w_A.weight, std=0.02)
        for w_B in self.w_Bs + self.mask_decoder_w_Bs:
            nn.init.trunc_normal_(w_B.weight, std=0.02)

    def save_lora_parameters(self, filename: str) -> None:
        # assert filename.endswith(".safetensors")

        num_image_encoder_layer = len(self.w_As)
        num_mask_decoder_layer = len(self.mask_decoder_w_As)

        a_tensors = {f"image_encoder_w_a_{i:03d}": self.w_As[i].weight for i in range(num_image_encoder_layer)}
        b_tensors = {f"image_encoder_w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_image_encoder_layer)}

        a_tensors.update(
            {f"mask_decoder_w_a_{i:03d}": self.mask_decoder_w_As[i].weight for i in range(num_mask_decoder_layer)})
        b_tensors.update(
            {f"mask_decoder_w_b_{i:03d}": self.mask_decoder_w_Bs[i].weight for i in range(num_mask_decoder_layer)})

        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        if not os.path.exists(filename):
            print(f"No LoRA checkpoint found at {filename}, initializing parameters")
            self.reset_parameters()
            return

        if filename.endswith(".safetensors"):
            self.load_from_safetensors(filename)
        elif filename.endswith(".pth"):
            self.load_from_pth(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .safetensors or .pth")

    def load_from_safetensors(self, filename: str) -> None:
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"image_encoder_w_a_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key)
                    w_A_linear.weight.data.copy_(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"image_encoder_w_b_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key)
                    w_B_linear.weight.data.copy_(saved_tensor)

            for i, w_A_linear in enumerate(self.mask_decoder_w_As):
                saved_key = f"mask_decoder_w_a_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key)
                    w_A_linear.weight.data.copy_(saved_tensor)

            for i, w_B_linear in enumerate(self.mask_decoder_w_Bs):
                saved_key = f"mask_decoder_w_b_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key)
                    w_B_linear.weight.data.copy_(saved_tensor)

    def load_from_pth(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location='cpu')

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_As.{i}.weight"
            if saved_key in state_dict:
                w_A_linear.weight.data.copy_(state_dict[saved_key])

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_Bs.{i}.weight"
            if saved_key in state_dict:
                w_B_linear.weight.data.copy_(state_dict[saved_key])

        for i, w_A_linear in enumerate(self.mask_decoder_w_As):
            saved_key = f"mask_decoder_w_As.{i}.weight"
            if saved_key in state_dict:
                w_A_linear.weight.data.copy_(state_dict[saved_key])

        for i, w_B_linear in enumerate(self.mask_decoder_w_Bs):
            saved_key = f"mask_decoder_w_Bs.{i}.weight"
            if saved_key in state_dict:
                w_B_linear.weight.data.copy_(state_dict[saved_key])

        print(f"Loaded LoRA parameters from {filename}")

