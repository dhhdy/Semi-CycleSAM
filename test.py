import os
import argparse
import torch

from test_util import test_all_case
from networks.unet_3D import unet_3D
from test_util import build_model
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='', help='model_name')
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
    save_mode_path = os.path.join('')
    save_mode_path_sam = os.path.join('')
    save_mode_path_sam2 = os.path.join('')
    net.load_state_dict(torch.load(save_mode_path))
    net_sam.load_state_dict(torch.load(save_mode_path_sam))
    net_sam2.load_state_dict(torch.load(save_mode_path_sam2))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    net_sam.eval()
    net_sam2.eval()
    avg_metric = test_all_case(net, net_sam, net_sam2, image_list, num_classes=num_classes,
                               patch_size=(128, 128, 128), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
