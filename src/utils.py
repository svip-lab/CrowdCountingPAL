import h5py
import torch
import shutil
import numpy as np

import glob
import os
import torch
import shutil
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.jet

def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='FCRN')

    parser.add_argument('--arch', default='CSRNet', type=str)
    parser.add_argument('--decoder', default='upproj', type=str)
    parser.add_argument('--resume',
                        # default='./result/UNet/run_1/checkpoint-20.pth.tar',
                        default=None,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler. '
                                                                   'See documentation of ReduceLROnPlateau.')

    parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--dataset', type=str, default="SHT_A")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--start_epoch', default=0, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--run_id', default=2021090110, type=int,
                        metavar='N', help='Run ID, YYYY+MM+DD+id')

    parser.add_argument('--train_json', default='train.json',metavar='TRAIN',
                    help='path to train json')
    parser.add_argument('--val_json', default='val.json', metavar='VAL',
                    help='path to val json')

    parser.add_argument('--gt_ratio', default='100', type=str, help='10, 20, 30, 40, 50, 60, 70, 100')
    parser.add_argument('--opt_type', default='Adam', type=str, help='Adam, SGD')

    parser.add_argument('--use_random_crop', dest='use_random_crop',help='Set to True to use randomly crop patches during training.', action='store_true',default=False)
    parser.add_argument('--not_use_adaLr', dest='not_use_adaLr',help='Set to True for forward network.', action='store_true',default=False)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)

    parser.add_argument('--is_eval', dest='is_eval',help='Set to True for forward network.', action='store_true',default=False)
    parser.add_argument('--epoch-st', dest='epoch_st', default=10, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--epoch-end', dest='epoch_end', default=20, type=int, help='number of data loading workers (default: 0)')

    args = parser.parse_args()
    return args

def get_output_directory(args):
    if args.resume:
        return os.path.dirname(args.resume)
    else:
        #save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = '../'
        #save_dir_root = os.path.join(save_dir_root, 'result', args.arch)
        #runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        #run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        #save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))

        save_dir_root = os.path.join(save_dir_root, 'exp', args.dataset ,args.arch)
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = args.run_id #int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
        return save_dir

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)




