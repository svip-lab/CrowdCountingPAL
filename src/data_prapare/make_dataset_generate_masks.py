import numpy as np
import scipy.ndimage
import scipy.io as scio
from PIL import Image
import copy
import os
import glob
import cv2
import h5py

def generate_data(is_train, is_max, is_rand, dataset_name, H_ratio=0.5, W_ratio=0.2, mask_ID='2020_12_16_v2_rand_'):

    raw_data_root = '/p300/2020-Counting/datasets/raw/' + dataset_name + '/train_data/'

    raw_matdata_root = raw_data_root + 'ground_truth/'
    raw_image_root = raw_data_root + 'images/'

    new_data_root = raw_data_root.replace('raw','masked_data')
    new_densitymap_root = new_data_root + 'ground_truth_density_map'
    new_maskmap_root = new_data_root + mask_ID + 'mask/'

    sigma = 15
    num_list = []

    gt_paths = []
    for gt_path in glob.glob(os.path.join(raw_matdata_root, '*.mat')):
        gt_paths.append(gt_path)

        gt_file = raw_matdata_root + 'GT_IMG_' + str(i) + '.mat'
        image_file = raw_image_root + 'IMG_' + str(i) + '.jpg'
        print(gt_file)

        #import pdb; pdb.set_trace()
        new_name = 'GT_IMG_' + str(i) + '.h5'

        gt = scio.loadmat(gt_file)
        if dataset_name == 'part_A_final' or dataset_name == 'part_B_final':
            gt = gt['image_info'][0][0][0][0][0]
        if dataset_name == 'UCF_QNRF':
            gt = gt['annPoints']

        image = Image.open(image_file)
        image = np.array(image)
        #H, W, C = image.shape
        H, W = image.shape[0], image.shape[1]

        gt_map = np.zeros((H,W))

        for j, (x,y) in enumerate(gt):
            if x>W or y>H:
                continue
            gt_map[int(y),int(x)] = 1

        sigma15, sigma20 = 15, 20
        density_map_sigma15 = scipy.ndimage.filters.gaussian_filter(gt_map, sigma15)
        density_map_sigma20 = scipy.ndimage.filters.gaussian_filter(gt_map, sigma20)

        mask = np.zeros((H,W))
        d_x = int(W* W_ratio)
        d_y = int(H* H_ratio)

        #import pdb;pdb.set_trace()
        if W_ratio < 1:
            if is_max:
                (pos_y, pos_x) = np.unravel_index(np.argmax(density_map_sigma15), density_map_sigma15.shape)
            if is_rand:
                #pos_y, pos_x = 0, 0
                if H-d_x-10<10:
                    pos_x = np.random.randint(0, W - d_x - 10)
                    pos_y = np.random.randint(0, H - d_y - 10)
                else:
                    pos_x = np.random.randint(10, W - d_x - 10)
                    pos_y = np.random.randint(10, H - d_y - 10)


            if pos_y >= image.shape[0] - d_y:
                pos_y = image.shape[0] - d_y
            if pos_y <= 0:
                pos_y = 0

            if pos_x >= image.shape[1] - d_x:
                pos_x = image.shape[1] - d_x
            if pos_x <= 0:
                pos_x = 0

            mask[pos_y:pos_y+d_y, pos_x:pos_x+d_x] = 1
        else:
            mask = np.ones((H,W))

        mask = mask > 0
        mask = np.array(mask + 0)

        mask_img_name = os.path.join(new_maskmap_root, new_name.replace('.h5','.png'))
        if not os.path.exists(new_maskmap_root):
            os.makedirs(new_maskmap_root)
        mask = mask * 255
        cv2.imwrite(mask_img_name, mask)

## Random Mask 10%
is_train, is_max, is_rand, dataset_name = True, False, True, 'part_A_final'
generate_data(is_train, is_max, is_rand, dataset_name)

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_B_final'
generate_data(is_train, is_max, is_rand, dataset_name)

is_train, is_max, is_rand, dataset_name = True, False, True, 'UCF_QNRF'
generate_data(is_train, is_max, is_rand, dataset_name)

'''
## Random mask 50%
H_ratio=0.7
W_ratio=0.7
mask_ID='2020_12_19_r50_rand_'

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_A_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_B_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'UCF_QNRF'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

## Random mask 25%
H_ratio=0.5
W_ratio=0.5
mask_ID='2020_12_19_r25_rand_'

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_A_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_B_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'UCF_QNRF'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

## Random mask 1%
H_ratio=0.1
W_ratio=0.1
mask_ID='2020_12_19_r01_rand_'

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_A_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'part_B_final'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)

is_train, is_max, is_rand, dataset_name = True, False, True, 'UCF_QNRF'
generate_data(is_train, is_max, is_rand, dataset_name, H_ratio, W_ratio, mask_ID)
'''
