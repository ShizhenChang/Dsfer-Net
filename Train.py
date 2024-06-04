# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:43:44 2022

@author: Shizhen Chang
"""
#For WHU-building dataset: 
   # More details can be found in "Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set"

import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from utils.DiceLoss import *
from Data.LEVIR.LEVIRDataSet import LEVIRDataSet
from Data.CDD.CDDDataSet import CDDDataSet
from Data.WHU_Building.WHUDataSet import WHUDataSet
from model.Networks import MyNet


DataName = {1:'LEVIR-CD', 2:'WHU_Building', 3:'CDD'}
name_classes = ['unchanged','changed']
epsilon = 1e-14


def main(args):
    if args.dataID == 1:
        train_list='./Data/LEVIR/train.txt'
        val_list='./Data/LEVIR/val.txt'  
        weight = np.array([1.04809706, 21.79129276])#LEVIR
        src_loader = data.DataLoader(
                        LEVIRDataSet(args.data_dir, train_list, max_iters=args.num_steps_stop*args.batch_size,set='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        val_loader = data.DataLoader(
                        LEVIRDataSet(args.data_dir, val_list,set='val'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)         
    elif args.dataID == 2:
        train_list='./Data/WHU_Building/train.txt'
        val_list='./Data/WHU_Building/val.txt'      
        weight = np.array([1.04715042, 22.20871691])#WHU_Building
        src_loader = data.DataLoader(
                        WHUDataSet(args.data_dir, train_list, max_iters=args.num_steps_stop*args.batch_size,set='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        val_loader = data.DataLoader(
                        WHUDataSet(args.data_dir, val_list,set='val'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)           
    elif args.dataID == 3:
        train_list='./Data/CDD/train.txt'
        val_list='./Data/CDD/val.txt'
        weight = np.array([1.13250422, 8.54692918])#CDD
        src_loader = data.DataLoader(
                        CDDDataSet(args.data_dir, train_list, max_iters=args.num_steps_stop*args.batch_size,set='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        val_loader = data.DataLoader(
                        CDDDataSet(args.data_dir, val_list,set='val'),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)                 

        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["CUDA_LAUNCH_BLOCKING"] = '0'
    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))

    cudnn.enabled = True
    cudnn.benchmark = True
    # Create network   
    model = MyNet(n_classes=args.num_classes, beta=args.beta, dim = args.project, numhead = args.numhead)
    model = model.cuda()
    #writer.add_graph(model, input_to_model = (torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)))

    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,5))
    F1_best = 0.85   
    cross_entropy_loss = nn.CrossEntropyLoss(torch.from_numpy(weight).cuda().float(),ignore_index=255)
    criterion1 = DiceLoss()
    criterion2 = DiceLoss()
    pool1 = nn.MaxPool2d(8, stride=8)
    pool2 = nn.MaxPool2d(16, stride=16)
    for batch_id, src_data in enumerate(src_loader):
        if batch_id==args.num_steps_stop:
            break
        tem_time = time.time()
        model.train()
        optimizer.zero_grad()
        
        imgAs, imgBs, labels, _, _ = src_data
        imgAs = imgAs.cuda()   
        imgBs = imgBs.cuda()
        label_4 = make_one_hot(pool1(labels.unsqueeze(1).float()).long(),2).cuda()
        label_5 = make_one_hot(pool2(labels.unsqueeze(1).float()) .long(),2).cuda()        
        labels = labels.cuda().long()

        pre_output, x_hp4, x_hp5 = model(imgAs, imgBs)          
        # CE Loss
        cross_entropy_loss_value = cross_entropy_loss(pre_output, labels)

        # Dice Loss
        consistent_loss1 = criterion1(x_hp4,label_4)
        consistent_loss2 = criterion2(x_hp5,label_5)

        _, predict_labels = torch.max(pre_output, 1)
        predict_labels = predict_labels.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        batch_oa = np.sum(predict_labels==labels)*1./len(labels.reshape(-1))

        total_loss = cross_entropy_loss_value + args.lam* 0.5* (consistent_loss1 + consistent_loss2)
        total_loss.backward()
        optimizer.step()

        hist[batch_id,0] = total_loss.item()
        hist[batch_id,1] = consistent_loss1
        hist[batch_id,2] = consistent_loss2
        hist[batch_id,3] = batch_oa
        hist[batch_id,-1] = time.time() - tem_time

        if (batch_id+1) % 100 == 0: 
            print('Iter %d/%d Time: %.2f Batch_OA = %.1f cross_entropy_loss = %.3f'%(batch_id+1,args.num_steps_stop,100*np.mean(hist[batch_id-99:batch_id,-1]),np.mean(hist[batch_id-99:batch_id,3])*100,np.mean(hist[batch_id-99:batch_id,0])))
           
        # evaluation per 500 iterations
        if (batch_id+1) % 500 == 0:            
            print('Validating.......')
            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
            for ID, batch in enumerate(val_loader):  
                image1, image2, label,_, name = batch
                label = label.squeeze().numpy()

                image1 = image1.float().cuda()
                image2 = image2.float().cuda()
                
                with torch.no_grad():
                    pred, _, _ = model(image1,image2)

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
            print('===>' + name_classes[1] + ' Precision: %.2f'%(P[1] * 100))
            print('===>' + name_classes[1] + ' Recall: %.2f'%(R[1] * 100))         
            print('===>' + name_classes[1] + ' F1: %.2f'%(F1[1] * 100))           
            print('===> IoU: %.2f, OA: %.2f'%(IoU[1]*100,OA[1]*100)) 
            
            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')                     
                model_name = str(args.dataID)+'_dataset_batchsize_'+str(args.batch_size)+'_beta_'+str(args.beta)+'_lam_'+str(args.lam)+'_lr_'+str(args.learning_rate)+'batch'+repr(batch_id+1)+'_F1_'+repr(int(F1[1]*10000))+'.pth'
                torch.save(model.state_dict(), os.path.join(
                    snapshot_dir, model_name))
        adjust_learning_rate(optimizer,args.learning_rate,batch_id,args.num_steps)

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
    parser.add_argument("--num_steps", type=int, default=30000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=15000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the model.")

    main(parser.parse_args())