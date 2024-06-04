
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import cv2 as cv

class WHUDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='train'):
        self.mean = [107.9191, 118.1207, 123.3417]
        self.std = [51.0839, 47.8814, 49.5432]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='train':
            for name in self.img_ids:
                img_file = data_dir + name
                img_file2 = img_file.replace('A', 'B')
                label_file = img_file.replace('A','label') 
                self.files.append({
                    "imgA": img_file,
                    "imgB": img_file2,
                    "label": label_file,
                    "name": name
                })
        elif set=='val':
            for name in self.img_ids:
                img_file = data_dir + name
                img_file2 = img_file.replace('A', 'B')
                label_file = img_file.replace('A','label') 
                self.files.append({
                    "imgA": img_file,
                    "imgB": img_file2,
                    "label": label_file,
                    "name": name
                })
        elif set=='test':
            for name in self.img_ids:
                img_file = data_dir + name
                img_file2 = img_file.replace('A', 'B')
                label_file = img_file.replace('A','label') 
                self.files.append({
                    "imgA": img_file,
                    "imgB": img_file2,
                    "label": label_file,
                    "name": name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        imgA = cv.imread(datafiles["imgA"])
        imgB = cv.imread(datafiles["imgB"])
        label = cv.imread(datafiles["label"])
        label =label[:,:,0]
        name = datafiles["name"]
            
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        label //= 255
                
        label = np.asarray(label, np.float32)
        imgA = imgA.transpose((-1, 0, 1))  
        imgB = imgB.transpose((-1, 0, 1))    
        #imgA = np.moveaxis(imgA, -1, 0)     
        #imgB = np.moveaxis(imgB, -1, 0)
        size = imgA.shape
        #print(size) #channel*height*weight
       
        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]
        return imgA.copy(), imgB.copy(), label.copy(), np.array(size), name
    
if __name__ == '__main__':
    
    train_dataset = WHUDataSet(data_dir='/root/datasets/Building_Change/',list_path='./train.txt')
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=False,pin_memory=True)
    class_label_num = np.zeros((1,2))
    channels_sumA,channel_squared_sumA,channels_sumB,channel_squared_sumB = 0,0,0,0
    num_batches = len(train_loader)
    index = 0
    for dataA,dataB,label,_,_ in train_loader:
        index += 1
        if index%1000==0:
           print(index,num_batches)
        channels_sumA += torch.mean(dataA,dim=[0,2,3])   
        channel_squared_sumA += torch.mean(dataA**2,dim=[0,2,3])       
        channels_sumB += torch.mean(dataB,dim=[0,2,3])   
        channel_squared_sumB += torch.mean(dataB**2,dim=[0,2,3])
        channels_sum = channels_sumA + channels_sumB
        channel_squared_sum = channel_squared_sumA + channel_squared_sumB         
        label = label.numpy()
        for i in range(2):
            class_label_num[0,i] += np.sum(label==i)

    weight = class_label_num/(num_batches*256*256)
    
    mean = (channels_sum)/(num_batches*2)
    std = ((channel_squared_sum) / (num_batches*2) - mean**2)**0.5 
    print(mean, std)
    #tensor([107.9191, 118.1207, 123.3417]) tensor([51.0839, 47.8814, 49.5432])
    print(weight) 
    #[[0.95497264 0.04502736]]
    print(1./weight) 
    #[[1.04715042 22.20871691]]
    
