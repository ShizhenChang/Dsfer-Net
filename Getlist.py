"""
Created on Fri Apr  8 16:43:04 2022

@author: Shizhen Chang
"""
import os
import numpy as np
import argparse
import random

DataName = {1:'LEVIR-CD', 2:'WHU_Building', 3:'CDD'}

def main(args):
    if args.dataID==1:
        rootDir_train = args.root_path +'/LEVIR-CD/tiles/train/A'
        rootDir_val = args.root_path +'/LEVIR-CD/tiles/val/A'
        rootDir_test = args.root_path +'/LEVIR-CD/tiles/test/A'
        f_train = open('./Data/LEVIR/train.txt', 'w')
        f_val = open('./Data/LEVIR/val.txt', 'w')
        f_test = open('./Data/LEVIR/test.txt', 'w')
        print('loading the LEVIR-CD dataset...OK')
        for dirpath, dirnames, filenames in os.walk(rootDir_train):
            for i in range(len(filenames)):
                f_train.writelines('LEVIR-CD/tiles/train/A/'+str(filenames[i])+'\n')
        for dirpath, dirnames, filenames in os.walk(rootDir_val):
            for i in range(len(filenames)):
                f_val.writelines('LEVIR-CD/tiles/val/A/'+str(filenames[i])+'\n')
        for dirpath, dirnames, filenames in os.walk(rootDir_test):
            for i in range(len(filenames)):
                f_test.writelines('LEVIR-CD/tiles/test/A/'+str(filenames[i])+'\n')

    if args.dataID==2:
        rootDir = args.root_path +'/WHU_Building/tiles/A'
        
        f_train = open('./Data/WHU_Building/train.txt', 'w')
        f_val = open('./Data/WHU_Building/val.txt', 'w')
        f_test = open('./Data/WHU_Building/test.txt', 'w')
        perc = [0.6, 0.1, 0.3]#the percentage of train/val/test
        for dirpath, dirnames, filenames in os.walk(rootDir):
            print('loading the WHU Building dataset...OK')
            random.shuffle(filenames)
        for i in range(len(filenames)): 
            if i<int(np.floor(len(filenames)*perc[0])):
                f_train.writelines('WHU_Building/tiles/A/'+str(filenames[i])+'\n')
            elif i<int(np.floor(len(filenames)*(perc[0]+perc[1]))):
                f_val.writelines(rootDir+'/'+str(filenames[i])+'\n')
                #print(filenames[i])
            else:
                f_test.writelines(rootDir+'/'+str(filenames[i])+'\n')
    if args.dataID==3:
        rootDir_train = args.root_path +'/CDD/train/A'
        rootDir_val = args.root_path +'/CDD/val/A'
        rootDir_test = args.root_path +'/CDD/test/A'
        f_train = open('./Data/CDD/train.txt', 'w')
        f_val = open('./Data/CDD/val.txt', 'w')
        f_test = open('./Data/CDD/test.txt', 'w')
        print('loading the CDD dataset...OK')
        for dirpath, dirnames, filenames in os.walk(rootDir_train):
            for i in range(len(filenames)):
                f_train.writelines('CDD/train/A/'+str(filenames[i])+'\n')
        for dirpath, dirnames, filenames in os.walk(rootDir_val):
            for i in range(len(filenames)):
                f_val.writelines('CDD/val/A/'+str(filenames[i])+'\n')
        for dirpath, dirnames, filenames in os.walk(rootDir_test):
            for i in range(len(filenames)):
                f_test.writelines('CDD/test/A/'+str(filenames[i])+'\n')


    f_train.close()
    f_val.close()
    f_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/root/datasets/Building_Change')
    parser.add_argument('--dataID', type=int, default=1)
    main(parser.parse_args())