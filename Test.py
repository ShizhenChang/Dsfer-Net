# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:10:55 2022

@author: Shizhen Chang
"""

import argparse
import numpy as np
import torch
from torch.utils import data
from utils.tools import *
from utils.DiceLoss import *
from Data.LEVIR.LEVIRDataSet import LEVIRDataSet
from Data.CDD.CDDDataSet import CDDDataSet
from Data.WHU_Building.WHUDataSet import WHUDataSet
from model.Networks import MyNet
import os
import torch.nn as nn

DataName = {1:'LEVIR-CD', 2:'WHU_Building', 3:'CDD'}
name_classes = ['unchanged','changed']
epsilon = 1e-14


def main(args):
    """Create the model and start the evaluation process."""
    if args.dataID == 1:
        test_list='./Data/LEVIR/test.txt' 
        test_loader = data.DataLoader(
                        LEVIRDataSet(args.data_dir, test_list,set='test'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        restore_from='./exp/LEVIR_F1_9094.pth' #path of stored model
        save_dir='./test/LEVIR/'

    elif args.dataID == 2:
        test_list='./Data/WHU_Building/test.txt'           
        test_loader = data.DataLoader(
                        WHUDataSet(args.data_dir, test_list, set='test'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)  
        restore_from='./exp/WHU_F1_9441.pth'
        save_dir='./test/WHU/'     

    elif args.dataID == 3:
        test_list='./Data/CDD/test.txt'
        test_loader = data.DataLoader(
                        CDDDataSet(args.data_dir, test_list, set='test'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        restore_from='./exp/CDD_F1_9524.pth'
        save_dir='./test/CDD/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(save_dir+'Evaluation.txt', 'w')
    
    model = MyNet(n_classes=args.num_classes, beta=args.beta, dim = args.project, numhead = args.numhead)

    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    for _, batch in enumerate(test_loader):  
        image1, image2, label,_, name = batch
        label = label.squeeze().numpy()
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        
        with torch.no_grad():
            pred, _, _ = model(image1, image2)
        _,pred = torch.max(nn.functional.softmax(pred,dim=1).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()        

        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    P =TP_all*1.0 / (TP_all + FP_all + epsilon)
    R = TP_all*1.0 / (TP_all + FN_all + epsilon)
    F1 = 2.0*P*R / (P + R + epsilon)
    IoU = TP_all*1.0 / (TP_all + FP_all + FN_all)
    OA = (TP_all+TN_all)*1.0 / n_valid_sample_all
    for i in range(args.num_classes):
        f.write('\n===>' + name_classes[i] + ' Precision: %.4f'%(P[i] * 100))
        print('===>' + name_classes[i] + ' Precision: %.4f'%(P[i] * 100))
        f.write('\n===>' + name_classes[i] + ' Recall: %.4f'%(R[i] * 100)) 
        print('===>' + name_classes[i] + ' Recall: %.4f'%(R[i] * 100))  
        f.write('\n===>' + name_classes[i] + ' F1: %.4f'%(F1[i] * 100))
        print('===>' + name_classes[i] + ' F1: %.4f'%(F1[i] * 100)) 
        f.write('\n===>' + name_classes[i] + 'IoU: %.4f'%(IoU[i] * 100))
        print('===>' + name_classes[i] + 'IoU: %.4f'%(IoU[i] * 100))
    f.write('\n===> mF1: %.4f'%(np.nanmean(F1) * 100))
    print('===> mF1: %.4f'%(np.nanmean(F1) * 100))
    f.write('\n===> OA: %.4f'%(OA[1] * 100))
    print('===> OA: %.4f'%(OA[1] * 100))
    f.write('\n===> mIoU: %.4f'%(np.mean(IoU) * 100))
    print('===> mIoU: %.4f'%(np.mean(IoU) * 100))

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)   
    parser.add_argument('--data_dir', type=str, default='/root/datasets/Building_Change/',
                        help="Root directory of the datasets.")
    parser.add_argument("--input_size", type=str, default='256,256',
                        help="width and height of input images.")    
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")   
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images in each batch.")
    parser.add_argument("--beta", type=float, default=512,
                        help="value of beta.")        
    parser.add_argument("--project", type=int, default=512,
                        help="the dimension of hopfield linear layer.")    
    parser.add_argument("--numhead", type=int, default=1,
                        help="the number of head for hopfield.")  
    parser.add_argument("--lam", type=float, default=1,
                        help="the proporation of const loss"
                        "LEVIR: 0.01; WHU: 1; CDD: 0.1.")                        
    parser.add_argument("--learning_rate", type=float, default=5e-05,
                        help="learning rate:5e-5--LEVIR&WHU; 1E-4--CDD.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--num_steps", type=int, default=30000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=15000,
                        help="number of training steps for early stopping.")



    main(parser.parse_args())