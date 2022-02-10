import sys
import os
import warnings
import numpy as np
import argparse
import json
import cv2
import time
import ipdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler

# Network
from model import CSRNet as net_PSL

# Dataset
import dataset

import utils
args = utils.parse_command()
print(args)

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def main():

    global args,best_prec1

    best_prec1 = 1e6
    args.decay         = 5*1e-4
    args.seed = int(time.time()) # args.manual_seed #int(time.time())

    train_list = json.load(open(args.train_json))
    val_list = json.load(open(args.val_json))
    torch.cuda.manual_seed(args.seed)

    #import pdb;pdb.set_trace()
    model = net_PSL()

    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    if args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.decay)
    if args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    # when training, use reduceLROnPlateau to reduce learning rate
    if not args.not_use_adaLr:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)

    # Save the config and results
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')


    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       # seen=model.seen,
                       seen=0,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)

    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        prec1, prec_net1, prec_net2, mse, mse_net1, mse_net2 = validate(val_loader, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))

        with open(best_txt, 'a+') as txtfile:
            txtfile.write(
            "epoch={}, mae={:.3f}, net1={:.3f}, net2={:.3f}, mse={:.3f}, net1={:.3f}, net2={:.3f} \n".
            format(epoch, prec1,prec_net1, prec_net2, mse, mse_net1, mse_net2))

        if is_best:
            #best_result = result
            with open(best_txt, 'a+') as txtfile:
                txtfile.write(
                    "Best epoch={}, mae={:.3f}, net1={:.3f}, net2={:.3f}, mse={:.3f}, net1={:.3f}, net2={:.3f} \n".
                        format(epoch, best_prec1,prec_net1, prec_net2, mse, mse_net1, mse_net2))
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model,
                'best_result': best_prec1,
                'optimizer': optimizer,
                'state': model.state_dict(),
            }, is_best, epoch, output_directory)

    
def train(train_loader, model, criterion, optimizer, epoch):

    criterion_lcm = nn.L1Loss().cuda()

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), optimizer.param_groups[0]['lr'] ))

    model.train()
    end = time.time()

    for i,(img, target, mask, target_20)in enumerate(train_loader):
        data_time.update(time.time() - end)

        #import pdb; pdb.set_trace()
        mask_clone = mask.clone()
        mask_clone = mask_clone.cuda()
        mask_clone = mask_clone.permute(0,2,1)

        mask[mask>0] = 1
        mask = mask.cuda()
        mask = mask.permute([0,2,1])

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        target_20 = target_20.type(torch.FloatTensor).cuda()
        target_20 = Variable(target_20)

        #import pdb; pdb.set_trace()
        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            output_1, output_2, _,_,_,_ = model(img, mask=None, target=None, train_flag=False)
            target_pred = (output_1 + output_2)/2
        output_1, output_2, latent_loss, _, diff_mean, diff_var = model(img, mask, target_pred)

        pred_loss_sum_net1_net2, pred_loss_sum_net2_net1 =0,0

        #  Net 1
        output_net1 = output_1[:,0,:,:]
        output_net1 = output_net1 * mask
        pred_loss_net1 = criterion(output_net1, target)


        # Net 2
        output_net2 = output_2[:,0,:,:]
        output_net2 = output_net2 * mask
        pred_loss_net2 = criterion(output_net2, target_20)


        #import pdb; pdb.set_trace()
        output_net1_sum = torch.sum(output_1,dim=[1,2,3])
        output_net2_sum = torch.sum(output_2, dim=[1,2,3])

        pred_loss_sum_net1_net2 = criterion_lcm(output_net1_sum, output_net2_sum.detach())
        pred_loss_sum_net2_net1 = criterion_lcm(output_net2_sum, output_net1_sum.detach())

        pred_loss = pred_loss_net1 + pred_loss_net2 + 0.1 * pred_loss_sum_net1_net2 + 0.1 * pred_loss_sum_net2_net1

        latent_loss_weight = 0.1
        latent_loss = latent_loss.mean()
        loss = pred_loss + latent_loss_weight * latent_loss

        dis_loss_weight = args.lambda_u * linear_rampup(epoch+i/(len(train_loader)))

        mean_ann, mean_unkn = diff_mean[0], diff_mean[1]
        var_ann, var_unkn = diff_var[0], diff_var[1]

        loss_mean = criterion(mean_ann.detach(), mean_unkn)
        loss_var = criterion(var_ann.detach(), var_unkn)
        dis_loss = loss_mean + loss_var

        loss = loss + dis_loss_weight * dis_loss

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):

    print ('begin val')
    model.eval()

    mae, mse = 0, 0
    mae_net1, mse_net1 = 0, 0
    mae_net2, mse_net2 = 0, 0

    for i,(img, target, mask, _) in enumerate(val_loader):
        h,w = img.shape[2:4]
        h_d = h/2
        w_d = w/2

        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())

        density_1, density_net2_1, _, _,_,_ = model(img_1)
        density_2, density_net2_2, _, _,_,_ = model(img_2)
        density_3, density_net2_3, _, _,_,_ = model(img_3)
        density_4, density_net2_4, _, _,_,_ = model(img_4)


        density_1 = density_1.data.cpu().numpy()
        density_2 = density_2.data.cpu().numpy()
        density_3 = density_3.data.cpu().numpy()
        density_4 = density_4.data.cpu().numpy()
        pred_sum_1 = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

        density_net2_1 = density_net2_1.data.cpu().numpy()
        density_net2_2 = density_net2_2.data.cpu().numpy()
        density_net2_3 = density_net2_3.data.cpu().numpy()
        density_net2_4 = density_net2_4.data.cpu().numpy()
        pred_sum_net2 = density_net2_1.sum()+density_net2_2.sum()+density_net2_3.sum()+density_net2_4.sum()

        pred_sum = (pred_sum_1 + pred_sum_net2) / 2

        mae_net1 += abs(pred_sum_1 - target.sum())
        mae_net2 += abs(pred_sum_net2 - target.sum())

        mse_net1 += (pred_sum_1 - target.sum()) ** 2
        mse_net2 += (pred_sum_net2 - target.sum()) ** 2

        mae += abs(pred_sum-target.sum())
        mse += (pred_sum - target.sum()) ** 2

    mae = mae/len(val_loader)
    mse = mse/len(val_loader)

    mae_net1 = mae_net1/len(val_loader)
    mse_net1 = mse_net1/len(val_loader)

    mae_net2 = mae_net2/len(val_loader)
    mse_net2 = mse_net2/len(val_loader)

    print(' * MAE {mae:.3f}, MAE Net1 {mae_net1:.3f}, MAE Net2 {mae_net2:.3f}, MSE {mse:.3f}'
              .format(mae=mae, mae_net1=mae_net1, mae_net2=mae_net2, mse=mse))

    return mae, mae_net1, mae_net2, np.sqrt(mse), np.sqrt(mse_net1), np.sqrt(mse_net2)

def eval():

    global args,best_prec1
    best_prec1 = 1e6
    args.decay         = 5*1e-4
    args.seed = int(time.time()) # args.manual_seed #int(time.time())

    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    #import pdb;pdb.set_trace()
    model = net_PSL()

    model = nn.DataParallel(model)
    model = model.cuda()

    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    for epoch in range(args.epoch_st, args.epoch_end):
        snapshot_fname = output_directory + '/checkpoint-' + str(epoch) + '.pth.tar'
        if not os.path.exists(snapshot_fname):
            continue #exit()
        print(snapshot_fname)

        ckpt = torch.load(snapshot_fname)
        model.load_state_dict(ckpt['state'])
        model.eval()

        mae, mse = 0, 0
        mae_net1, mse_net1 = 0, 0
        mae_net2, mse_net2 = 0, 0

        for i,(img, target, mask, _, img_path) in enumerate(val_loader):
            h,w = img.shape[2:4]
            h_d = h/2
            w_d = w/2

            img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
            img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
            img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
            img_4 = Variable(img[:,:,h_d:,w_d:].cuda())

            density_1, density_net2_1, _, _,_,_ = model(img_1)
            density_2, density_net2_2, _, _,_,_ = model(img_2)
            density_3, density_net2_3, _, _,_,_ = model(img_3)
            density_4, density_net2_4, _, _,_,_ = model(img_4)


            density_1 = density_1.data.cpu().numpy()
            density_2 = density_2.data.cpu().numpy()
            density_3 = density_3.data.cpu().numpy()
            density_4 = density_4.data.cpu().numpy()
            pred_sum_1 = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

            density_net2_1 = density_net2_1.data.cpu().numpy()
            density_net2_2 = density_net2_2.data.cpu().numpy()
            density_net2_3 = density_net2_3.data.cpu().numpy()
            density_net2_4 = density_net2_4.data.cpu().numpy()
            pred_sum_net2 = density_net2_1.sum()+density_net2_2.sum()+density_net2_3.sum()+density_net2_4.sum()

            pred_sum = (pred_sum_1 + pred_sum_net2) / 2

            mae_net1 += abs(pred_sum_1 - target.sum())
            mae_net2 += abs(pred_sum_net2 - target.sum())

        mse_net1 += (pred_sum_1 - target.sum()) ** 2
        mse_net2 += (pred_sum_net2 - target.sum()) ** 2

        mae += abs(pred_sum-target.sum())
        mse += (pred_sum - target.sum()) ** 2

    mae = mae/len(val_loader)
    mse = mse/len(val_loader)

    mae_net1 = mae_net1/len(val_loader)
    mse_net1 = mse_net1/len(val_loader)

    mae_net2 = mae_net2/len(val_loader)
    mse_net2 = mse_net2/len(val_loader)

    print(' * MAE {mae:.3f}, MAE Net1 {mae_net1:.3f}, MAE Net2 {mae_net2:.3f}, MSE {mse:.3f}'
              .format(mae=mae, mae_net1=mae_net1, mae_net2=mae_net2, mse=mse))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    if args.is_eval:
        eval()
    else:
        main()
