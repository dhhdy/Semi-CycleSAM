import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from test_util_sam import var_all_case_LA, var_all_case_LA_single
import segmentation_models_pytorch_3d as smp
from torch.autograd import Variable
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.loss import KDLoss
from skimage.measure import label
from networks.vnet import VNet
from segment_anything.build_sam3D import sam_model_registry3D
from LoRA_SAM import LoRA_Sam3D
from torch.cuda.amp import autocast
from collections import deque
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data1/data/LA/2018LA_Seg_Training Set/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/data1/data/GD_master/h5/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/data1/data/BraTS2019/BraTS2019/data/', help='BraTS2019')
parser.add_argument('--exp', type=str,  default='mySAM', help='model_name')
parser.add_argument('--model', type=str,  default='VNet', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=20000, help='maximum epoch number to train')
parser.add_argument('--pre_max_iteration', type=int,  default=10000, help='maximum pre-train iteration to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--lora_lr', type=float,  default=0.0005, help='maximum epoch number to train lora')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--label_num', type=str,  default=4, help='number of labeled cases')
parser.add_argument('--clip', type=float,  default=2.0, help='number of labeled cases')
# parser.add_argument('--max_samples', type=str,  default=163, help='number of training set')
parser.add_argument('--max_samples', type=str,  default=80, help='number of training set')
parser.add_argument('--model_type', type=str,  default='vit_b_ori', help='model_type')
parser.add_argument('--checkpoint_path1', type=str,  default='/data1/data/LA/sam_med3d_turbo.pth', help='SAM checkpoint path1')
parser.add_argument('--checkpoint_path2', type=str,  default='/data1/data/LA/sam_med3d.pth', help='SAM checkpoint path2')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')  ######
args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
lora_lr = args.lora_lr
labeled_bs = args.labeled_bs
pre_max_iterations = args.pre_max_iteration
device = args.device
clip = args.clip
consistency_rampup = args.consistency_rampup
num_classes = 1
patch_size = (128, 128, 128)
CE = torch.nn.BCEWithLogitsLoss()
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net(net, path, is_ema):
    state = torch.load(str(path))
    net.load_state_dict(state)
    if is_ema:
        for param in net.parameters():
            param.detach_()

def mse_loss(input1, input2):
    input1 = torch.sigmoid(input1)
    input2 = torch.sigmoid(input2)
    return torch.mean((input1 - input2)**2)

def get_cut_mask_VNet(out, nms=0):
    probs = torch.sigmoid(out)[:,0,:,:,:] # one class
    masks = (probs > 0.5)
    masks = masks.contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks.long()

def get_cut_mask_SAM(out, nms=0):
    probs = torch.sigmoid(out)
    masks = (probs>0.5)
    masks = masks.contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks.long()

def loss_diff(u_prediction_1, u_prediction_2):
    u_1, u_2 = u_prediction_1, torch.sigmoid(u_prediction_2)
    loss_b = CE(u_1.clamp(1e-8, 1 - 1e-7),
                                 Variable(u_2.float(), requires_grad=False))
    return loss_b

def entropy_loss(p):
    # p N*C*W*H*D
    p = torch.sigmoid(p)  # 输出形状: [N, 1, W, H, D]

    # 计算二元熵: -p*log(p) - (1-p)*log(1-p)
    entropy = -p * torch.log(p + 1e-6) - (1 - p) * torch.log(1 - p + 1e-6)
    return torch.mean(entropy)

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def dice_loss(predict, target, weight=None, epsilon=1e-5):
    # predict = predict[:,0,:,:,:]
    num = predict.size(0)
    # pred不需要转bool变量，如https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py#L44
    # soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
    pred = torch.sigmoid(predict)
    pred = torch.cat([1-pred, pred], dim=1).view(num, 2, -1)
    targ = target.view(num, 1, -1)
    targ = to_one_hot(targ.type(torch.long), 2).type(torch.float16)
    intersection = pred * targ
    union = pred + targ
    # print(intersection.shape, union.shape)
    if weight is None:
        intersection = intersection.view(num, 2, -1).sum(2)
        union = union.view(num, 2, -1).sum(2)
    else:
        weight = weight.view(num, 1, -1)
        intersection = (intersection.view(num, 2, -1) * weight).sum(2)
        union = (union.view(num, 2, -1) * weight).sum(2)
    dice = (2 * intersection + epsilon) / (union + epsilon)
    score = 1 - dice.mean()
    return score


def get_max_var_area(img, model, x, y, z, T=10):

    xx,yy,zz = 0,0,0
    preda = []
    for _ in range(T):
        noise1 = torch.clamp(torch.randn_like(
            img) * 0.1, -0.05, 0.05)
        img_a_n = img + noise1
        pred_a = model(img_a_n)
        preda.append(pred_a.unsqueeze(0))
    outputs_a = torch.cat(preda, dim=0)
    var_a = outputs_a.var(dim=(0,1))
    var_a = var_a / (var_a.max() + 1)
    var_3d = var_a[0]  # 假设var_a的形状为 (1, 1, 128, 128, 128)
    s = var_3d.cumsum(dim=0).cumsum(dim=1).cumsum(dim=2)
    # 计算所有可能的立方体和
    # 切片处理，确保不越界
    if x > 0 and y > 0 and z > 0 and s.shape[0] >= x and s.shape[1] >= y and s.shape[2] >= z:
        val = (
                s[x:, y:, z:]
                - s[:-x, y:, z:]
                - s[x:, :-y, z:]
                - s[x:, y:, :-z]
                + s[:-x, :-y, z:]
                + s[:-x, y:, :-z]
                + s[x:, :-y, :-z]
                - s[:-x, :-y, :-z]
        )
        # 找到最大值及其索引
        if val.numel() > 0:
            max_val, flat_idx = torch.max(val.view(-1), dim=0)
            # (i,j,k)为左上角的最大和
            i, j, k = np.unravel_index(flat_idx.cpu().numpy(), val.shape)
            xx,yy,zz=i,j,k
    return xx,yy,zz

def context_mask(img, img_a, model, mask_ratio=2/3):
    _, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    batch_size = img_a.shape[0]
    loss_mask = torch.ones((batch_size, img_x, img_y, img_z), dtype=torch.float16).cuda()
    mask = torch.ones((img_x, img_y, img_z), dtype=torch.float16).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)

    # 对uimg使用蒙特卡罗dropout 计算var不确定性方差
    w,h,z = get_max_var_area(img, model, patch_pixel_x, patch_pixel_y, patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask1 = loss_mask / 2. + 0.5
    loss_mask2 = (1 - loss_mask) / 2. + 0.5
    return mask.long(), loss_mask1, loss_mask2


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).to(device)

def build_model(opt):
    checkpoint_path1 = args.checkpoint_path1
    checkpoint_path2 = args.checkpoint_path2
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if opt == 1:
        model_dict = torch.load(checkpoint_path1, map_location=device)
    else:
        model_dict = torch.load(checkpoint_path2, map_location=device)
    state_dict = model_dict['model_state_dict']
    sam_model.load_state_dict(state_dict)
    sam_model = LoRA_Sam3D(sam_model, 4).to(device)

    return sam_model

def create_model(nclass=1, ema=False):
    # Network definition
    net = smp.Unet(encoder_name='resnext101_32x4d', encoder_weights='swsl', in_channels=1, classes=1)
    model = net.to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)  #todo is or no
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def finetune_model_predict3D(img3D, stage, sam_model_tune, device='cuda'):

    image_embedding = sam_model_tune.image_encoder(img3D.to(device))  # [N,384,8,8,8]
    sparse_embeddings, dense_embeddings, _, kl_loss = sam_model_tune.auto_prompt(image_embedding)

    low_res_masks, _ = sam_model_tune.mask_decoder(
        image_embeddings=image_embedding.to(device),
        image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    prev_masks = F.interpolate(low_res_masks, size=(img3D.shape[-3:]), mode='trilinear', align_corners=False)
    medsam_seg_prob = prev_masks
    total_kl_loss = kl_loss
    return medsam_seg_prob, total_kl_loss

def pre_train(args, snapshot_path):
    model = create_model()
    sam_model_tune = build_model(1)
    sam_model_tune2 = build_model(2)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.AdamW(model.parameters(), lr=lora_lr, betas=(0.9, 0.999), weight_decay=0.1)
    optimizer_sam = optim.AdamW(sam_model_tune.parameters(), lr=lora_lr, betas=(0.9, 0.999), weight_decay=0.1)
    optimizer_sam2 = optim.AdamW(sam_model_tune2.parameters(), lr=lora_lr, betas=(0.9, 0.999), weight_decay=0.1)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.label_num
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    sam_model_tune.train()
    sam_model_tune2.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice1, best_dice2, best_dice3 = 0, 0, 0
    sub_bs = int(args.labeled_bs//2)
    iters_to_accumulate = 1
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    scaler = torch.cuda.amp.GradScaler()
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            with autocast():
                volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                # print(volume_batch.shape)
                if sub_bs > 0:
                    img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
                    lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
                else:
                    img_a, img_b = volume_batch[0].unsqueeze(0), volume_batch[0].unsqueeze(0)
                    lab_a, lab_b = label_batch[0].unsqueeze(0), label_batch[0].unsqueeze(0)
                with torch.no_grad():
                    img_mask, _, __ = context_mask(volume_batch, img_a, model)  # random mask?
                volume_batch = img_a * img_mask + img_b * (1 - img_mask)
                label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

                outputs = model(volume_batch)
                samseg_mask, kl_loss = finetune_model_predict3D(
                    volume_batch, None, sam_model_tune,
                    device=device)
                samseg_mask2, kl_loss2 = finetune_model_predict3D(
                    volume_batch, None, sam_model_tune2,
                    device=device)
                loss_sam_dice = dice_loss(samseg_mask, label_batch) + dice_loss(samseg_mask2, label_batch)
                loss_dice = dice_loss(outputs, label_batch)
                loss1 = loss_dice
                loss2 = loss_sam_dice + 0.1 * kl_loss + 0.1 * kl_loss2
                iter_num += 1
                total_loss = (loss1 + loss2) / iters_to_accumulate
            scaler.scale(total_loss).backward()
            if (iter_num + 1) % iters_to_accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scaler.unscale_(optimizer_sam)
                torch.nn.utils.clip_grad_norm_(sam_model_tune.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam)
                scaler.update()
                optimizer_sam.zero_grad()
                scaler.unscale_(optimizer_sam2)
                torch.nn.utils.clip_grad_norm_(sam_model_tune2.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam2)
                scaler.update()
                optimizer_sam2.zero_grad()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f'%(iter_num, loss1.item(), loss_dice.item()))
            logging.info('iteration %d : loss_sam: %03f, loss_sam_dice: %03f, loss_sam_kl: %03f'%(iter_num, loss2.item(), loss_sam_dice.item(), kl_loss.item()))
            if iter_num % 200 == 0:
                model.eval()
                sam_model_tune.eval()
                sam_model_tune2.eval()
                dice_sample = var_all_case_LA_single(model, opt=1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice1:
                    best_dice1 = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice.pth'.format(iter_num))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                dice_sample = var_all_case_LA_single(sam_model_tune2, opt=2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice2:
                    best_dice2 = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice.pth'.format(iter_num))
                    save_best_path = os.path.join(snapshot_path,'{}_best_sammodel2.pth'.format(args.model))
                    torch.save(sam_model_tune2.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                dice_sample = var_all_case_LA_single(sam_model_tune, opt=2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice3:
                    best_dice3 = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice.pth'.format(iter_num))
                    save_sambest_path = os.path.join(snapshot_path,'{}_best_sammodel.pth'.format(args.model))
                    torch.save(sam_model_tune.state_dict(), save_sambest_path)
                    logging.info("save best model to {}".format(save_mode_path))
                model.train()
                sam_model_tune.train()
                sam_model_tune2.train()
            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break


def self_train(pre_snapshot_path, self_snapshot_path):

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(args.label_num))
    unlabeled_idxs = list(range(args.label_num, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    ########  Load SAM ########
    sam_model_tune = build_model(1)
    sam_model_tune2 = build_model(2)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_sam = optim.AdamW(sam_model_tune.parameters(), lr=lora_lr, betas=(0.9, 0.999), weight_decay=0.1)
    optimizer_sam2 = optim.AdamW(sam_model_tune2.parameters(), lr=lora_lr, betas=(0.9, 0.999), weight_decay=0.1)
    rkd_loss = KDLoss()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    pretrained_sammodel = os.path.join(pre_snapshot_path, f'{args.model}_best_sammodel.pth')
    pretrained_sammodel2 = os.path.join(pre_snapshot_path, f'{args.model}_best_sammodel2.pth')
    load_net(ema_model, pretrained_model, True)
    load_net(model, pretrained_model, False)
    load_net(sam_model_tune, pretrained_sammodel, False)
    load_net(sam_model_tune2, pretrained_sammodel2, False)
    iter_num = 0
    best_dice1, best_dice2, best_dice3 = 0, 0, 0
    max_epoch = max_iterations//len(trainloader)+1
    sub_bs = int(args.labeled_bs//2)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    ema_model.train()
    sam_model_tune.train()
    sam_model_tune2.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            with autocast():
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                if sub_bs > 0:
                    img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
                    lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
                    unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[
                                                                                               args.labeled_bs + sub_bs:]
                    unimg = volume_batch[args.labeled_bs:]
                else:
                    img_a, img_b = volume_batch[0].unsqueeze(0), volume_batch[0].unsqueeze(0)
                    lab_a, lab_b = label_batch[0].unsqueeze(0), label_batch[0].unsqueeze(0)
                    unimg_a, unimg_b = volume_batch[0].unsqueeze(0), volume_batch[args.labeled_bs + sub_bs:]

                with torch.no_grad():  # create plabel
                    prev_label1_a = ema_model(unimg_a)
                    prev_label1_b = ema_model(unimg_b)
                    plab_a = get_cut_mask_VNet(prev_label1_a, nms=0)  # 伪标签ab
                    plab_b = get_cut_mask_VNet(prev_label1_b, nms=0)
                    predW_a, predH_a = plab_a.max(dim=1, keepdim=True)[0], plab_a.max(dim=2, keepdim=True)[0]
                    bboxWH_a = torch.minimum(predW_a, predH_a)
                    predW_b, predH_b = plab_b.max(dim=1, keepdim=True)[0], plab_b.max(dim=2, keepdim=True)[0]
                    bboxWH_b = torch.minimum(predW_b, predH_b)
                    img_mask, loss_mask_l, loss_mask_u = context_mask(unimg, img_a, ema_model)  # loss_mask center is 0
                mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
                mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
                mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
                mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
                bboxl_lab = lab_a * img_mask + bboxWH_a * (1 - img_mask)
                bboxu_lab = bboxWH_b * img_mask + lab_b * (1 - img_mask)

                # sam
                if iter_num % 3 == 0:
                    outputs_l = model(mixl_img)
                    outputs_u = model(mixu_img)

                    loss_u = dice_loss(outputs_u, mixu_lab, loss_mask_u)
                    loss_l = dice_loss(outputs_l, mixl_lab, loss_mask_l)
                    outputs_sam_l, kl_loss1 = finetune_model_predict3D(mixl_img, 2, sam_model_tune, device=device)
                    outputs_sam_u, kl_loss2 = finetune_model_predict3D(mixu_img, 2, sam_model_tune, device=device)

                    predW_a, predH_a = outputs_sam_l.max(dim=2, keepdim=True)[0], outputs_sam_l.max(dim=3, keepdim=True)[0]
                    predWH_sam_l = torch.minimum(predW_a, predH_a)
                    predW_a, predH_a = outputs_sam_u.max(dim=2, keepdim=True)[0], outputs_sam_u.max(dim=3, keepdim=True)[0]
                    predWH_sam_u = torch.minimum(predW_a, predH_a)
                    bboxWH_sam_u = predWH_sam_u * img_mask + outputs_sam_u * (1 - img_mask)
                    bboxWH_sam_l = outputs_sam_l * img_mask + predWH_sam_l * (1 - img_mask)
                    loss_sam_u = dice_loss(bboxWH_sam_u, bboxu_lab, loss_mask_u)
                    loss_sam_l = dice_loss(bboxWH_sam_l, bboxl_lab, loss_mask_l)
                    sam_consistency = rkd_loss(outputs_l, outputs_sam_l.clone().detach()) + rkd_loss(outputs_u, outputs_sam_u.clone().detach())
                    sam_con_loss = sam_consistency
                    loss = loss_u + loss_l + sam_con_loss
                    loss_sam = loss_sam_l + loss_sam_u + 0.1 * kl_loss1 + 0.1 * kl_loss2
                    total_loss = loss + loss_sam

                #sam2
                elif iter_num % 3 == 1:
                    outputs_l = model(mixl_img)
                    outputs_u = model(mixu_img)

                    loss_u = dice_loss(outputs_u, mixu_lab, loss_mask_u)
                    loss_l = dice_loss(outputs_l, mixl_lab, loss_mask_l)
                    outputs_sam_l_, kl_loss1 = finetune_model_predict3D(mixl_img, 2, sam_model_tune2, device=device)
                    outputs_sam_u_, kl_loss2 = finetune_model_predict3D(mixu_img, 2, sam_model_tune2, device=device)

                    # bbox loss
                    predW_a, predH_a = outputs_sam_l_.max(dim=2, keepdim=True)[0], outputs_sam_l_.max(dim=3, keepdim=True)[0]
                    predWH_sam_l = torch.minimum(predW_a, predH_a)
                    predW_a, predH_a = outputs_sam_u_.max(dim=2, keepdim=True)[0], outputs_sam_u_.max(dim=3, keepdim=True)[0]
                    predWH_sam_u = torch.minimum(predW_a, predH_a)
                    bboxWH_sam_u = predWH_sam_u * img_mask + outputs_sam_u_ * (1 - img_mask)
                    bboxWH_sam_l = outputs_sam_l_ * img_mask + predWH_sam_l * (1 - img_mask)
                    loss_sam_u = dice_loss(bboxWH_sam_u, bboxu_lab, loss_mask_u)
                    loss_sam_l = dice_loss(bboxWH_sam_l, bboxl_lab, loss_mask_l)

                    # consist loss
                    sam_consistency = rkd_loss(outputs_l, outputs_sam_l_.clone().detach()) + rkd_loss(outputs_u, outputs_sam_u_.clone().detach())
                    sam_con_loss = sam_consistency
                    loss = loss_u + loss_l + sam_con_loss
                    loss_sam = loss_sam_l + loss_sam_u + 0.1 * kl_loss1 + 0.1 * kl_loss2
                    total_loss = loss + loss_sam

                else:
                    outputs_sam_l, kl_loss1 = finetune_model_predict3D(mixl_img, 2, sam_model_tune, device=device)
                    outputs_sam_u, kl_loss2 = finetune_model_predict3D(mixu_img, 2, sam_model_tune, device=device)

                    outputs_sam_l_, kl_loss1_ = finetune_model_predict3D(mixl_img, 2, sam_model_tune2, device=device)
                    outputs_sam_u_, kl_loss2_ = finetune_model_predict3D(mixu_img, 2, sam_model_tune2, device=device)

                    sam_con_loss = loss_diff(outputs_sam_l, outputs_sam_l_.clone().detach()) + loss_diff(outputs_sam_u, outputs_sam_u_.clone().detach())+\
                                   loss_diff(outputs_sam_l_, outputs_sam_l.clone().detach()) + loss_diff(outputs_sam_u_, outputs_sam_u.clone().detach())
                    beta = get_current_consistency_weight(epoch_num)
                    # print(epoch_num, beta)
                    total_loss = (beta*sam_con_loss + 0.1*(kl_loss1+kl_loss1_+kl_loss2+kl_loss2_))

            scaler.scale(total_loss).backward()

        # update prams
            if iter_num % 3 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scaler.unscale_(optimizer_sam)
                torch.nn.utils.clip_grad_norm_(sam_model_tune.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam)
                scaler.update()
                optimizer_sam.zero_grad()
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            elif iter_num % 3 == 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scaler.unscale_(optimizer_sam2)
                torch.nn.utils.clip_grad_norm_(sam_model_tune2.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam2)
                scaler.update()
                optimizer_sam2.zero_grad()
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            else:
                scaler.unscale_(optimizer_sam)
                torch.nn.utils.clip_grad_norm_(sam_model_tune.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam)
                scaler.update()
                optimizer_sam.zero_grad()
                scaler.unscale_(optimizer_sam2)
                torch.nn.utils.clip_grad_norm_(sam_model_tune2.parameters(), max_norm=2.0)
                scaler.step(optimizer_sam2)
                scaler.update()
                optimizer_sam2.zero_grad()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations)
            lora_lr_ = lora_lr * (1.0 - iter_num / max_iterations)
            for param_group in optimizer_sam.param_groups:
                param_group['lr'] = lora_lr_
            for param_group in optimizer_sam2.param_groups:
                param_group['lr'] = lora_lr_
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                sam_model_tune.eval()
                sam_model_tune2.eval()
                dice_sample = var_all_case_LA_single(model, opt=1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice1:
                    best_dice1 = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}.pth'.format(iter_num))
                    save_best_path = os.path.join(self_snapshot_path,'VNet_best_model.pth')
                    torch.save(model.state_dict(), save_best_path)

                    logging.info("best dice {}, save best model to {}".format(best_dice1, save_mode_path))
                dice_sample = var_all_case_LA_single(sam_model_tune, opt=2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice2:
                    best_dice2 = round(dice_sample, 4)
                    save_best_path_sam = os.path.join(self_snapshot_path,'SAM_best_sam_model.pth')

                    torch.save(sam_model_tune.state_dict(), save_best_path_sam)
                    logging.info("best dice {}, save best model to {}".format(best_dice2, save_mode_path))

                dice_sample = var_all_case_LA_single(sam_model_tune2, opt=2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice3:
                    best_dice3 = round(dice_sample, 4)
                    save_best_path_sam = os.path.join(self_snapshot_path,'SAM_best_sam_model2.pth')
                    torch.save(sam_model_tune2.state_dict(), save_best_path_sam)
                    logging.info("best dice {}, save best model to {}".format(best_dice3, save_mode_path))
                model.train()
                sam_model_tune.train()
                sam_model_tune2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

    # writer.close()
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_snapshot_path = "./model/mySAM/LA_{}_{}_labeled/pre_train".format(args.exp, args.label_num)
    self_snapshot_path = "./model/mySAM/LA_{}_{}_labeled/self_train".format(args.exp, args.label_num)
    # pre_train(args, snapshot_path=pre_snapshot_path)
    self_train(pre_snapshot_path, self_snapshot_path)
