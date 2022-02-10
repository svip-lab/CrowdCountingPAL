import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import ipdb

import utils
args = utils.parse_command()

def load_data_masked_data(img_path, train = True):

    if train:
        #import pdb; pdb.set_trace()
        gt_ratio = 'ground_truth_density_map_sigma15'
        gt_path = img_path.replace('.jpg','.h5').replace('images', gt_ratio).replace('raw','masked_data').replace('IMG','GT_IMG')

        gt_ratio_20 = 'ground_truth_density_map_sigma20'
        gt_path_20 = img_path.replace('.jpg','.h5').replace('images',gt_ratio_20).replace('raw','masked_data').replace('IMG','GT_IMG')

        mask_ratio = args.gt_ratio + '_mask'
        mask_path = img_path.replace('images',mask_ratio).replace('raw','masked_data').replace('IMG','GT_IMG').replace('.jpg','.png')

    else:
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
        mask_path = img_path
        gt_path_20 = img_path.replace('.jpg','.h5').replace('images','ground_truth')

        gt_ratio = 'ground_truth_density_map_sigma15'
        gt_ratio_20 = 'ground_truth_density_map_sigma20'
        mask_ratio = args.gt_ratio + '_mask'

    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])
    mask = Image.open(mask_path).convert('L')

    gt_file_20 = h5py.File(gt_path_20,'r')
    target_20 = np.asarray(gt_file['density'])

    #import pdb;pdb.set_trace()
    cnt = 0
    if train:
        while(1):
            ratio = 0.5
            crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
            rdn_value = random.random()
            if rdn_value<0.25:
                dx = 0
                dy = 0
            elif rdn_value<0.5:
                dx = int(img.size[0]*ratio)
                dy = 0
            elif rdn_value<0.75:
                dx = 0
                dy = int(img.size[1]*ratio)
            else:
                dx = int(img.size[0]*ratio)
                dy = int(img.size[1]*ratio)

            if args.use_random_crop:
                dx = random.randint(0,int(img.size[0]*ratio))
                dy = random.randint(0,int(img.size[1]*ratio))

            img_rt = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            target_rt = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
            target_20_rt = target_20[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
            mask_rt = mask.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))

            if random.random()>0.8:
                target_rt = np.fliplr(target_rt)
                target_20_rt = np.fliplr(target_20_rt)
                mask_rt = mask_rt.transpose(Image.FLIP_LEFT_RIGHT)
                img_rt = img_rt.transpose(Image.FLIP_LEFT_RIGHT)

            if args.dataset == 'SHT_A':
                break_cnt = 1500 * 255
            if args.dataset == 'SHT_B':
                break_cnt = 3000 * 255

            if np.array(mask_rt).sum()>break_cnt:
                break

            cnt += 1

            if cnt > 100:
                exit()
                import pdb; pdb.set_trace()
            #print('cnt:', cnt, np.array(mask_rt).sum(),img_path)

        target_rt = cv2.resize(target_rt,(int(target_rt.shape[1]/8),int(target_rt.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
        target_20_rt = cv2.resize(target_20_rt,(int(target_20_rt.shape[1]/8),int(target_20_rt.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    if train:
        #import pdb;pdb.set_trace()
        mask_rt = np.array(mask_rt)
        mask_rt = cv2.resize(mask_rt,(int(mask_rt.shape[0]/8),int(mask_rt.shape[1]/8)),interpolation = cv2.INTER_CUBIC)
        #print(mask_rt.sum())
        #print(img_rt.size, target_rt.shape)
        return img_rt,target_rt, mask_rt, target_20_rt
    else:
        target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
        if args.is_eval:
            return img, target, target, target, img_path
        else:
            return img, target, target, target

