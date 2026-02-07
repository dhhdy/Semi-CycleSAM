import os
import argparse
import torch
from networks.unet_3D import unet_3D
import segmentation_models_pytorch_3d as smp
from test_util_sam import test_all_case_mysam
from train_LA_semisam_mt_lora_amp_STTS_diedai import build_model
from networks.unet_3D import unet_3D
from segment_anything.build_sam3D import sam_model_registry3D
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data1/data/LA/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='MCUD', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/data1/data/"
num_classes = 1

with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]

def test_calculate_metric():

    net = unet_3D(n_classes=1, in_channels=1).cuda()
    net_sam, _ = build_model()
    net_sam2, __ = build_model()
    test_save_path = "../model/prediction/" + FLAGS.model
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    save_mode_path = '../UNet_best_model.pth'
    save_mode_path_sam = '../SAM_best_sam_model.pth'
    save_mode_path_sam2 = '../SAM_best_sam_model2.pth'
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to('cuda')
    model_dict = torch.load(save_mode_path_sam, map_location='cuda')
    state_dict = model_dict['model_state_dict']
    sam_model_tune.load_state_dict(state_dict)
    sam_model_tune, lora_params = build_model()
    sam_model_tune.load_state_dict(torch.load(save_mode_path_sam))
    net.load_state_dict(torch.load(save_mode_path))
    net_sam.load_state_dict(torch.load(save_mode_path_sam))
    net_sam2.load_state_dict(torch.load(save_mode_path_sam2))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    net_sam.eval()
    net_sam2.eval()
    avg_metric = test_all_case_mysam(net, net_sam, net_sam2, image_list, num_classes=num_classes,
                               patch_size=(128, 128, 128), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
