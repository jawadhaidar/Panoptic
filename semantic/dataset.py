import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import os 
import cv2 as cv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import re


#thing+stuff labels cover indices 0-181 and 255 indicates the 'unlabeled' or void class.
class Coco_Stuff_things(Dataset):
    def __init__(self,transform=None):
        self.img_path=r"C:\Users\User\Desktop\AUB position\datasets\coco\val2017\val2017"
        self.mask_path=r"C:\Users\User\Desktop\AUB position\datasets\coco\stuffthingmaps_trainval2017\val2017"
        self.img_names_list=os.listdir( self.img_path)
        self.mask_names_list=os.listdir( self.mask_path)
        self.num_imgs=len(self.img_names_list)
        self.transform=transform
        #print(len( self.mask_names_list))
        #print(len( self.img_names_list))

    def __len__(self):
        return self.num_imgs 

    def __getitem__(self, index):
        img=cv.cvtColor(cv.imread(os.path.join(self.img_path,self.img_names_list[index])), cv.COLOR_BGR2RGB) 
        mask=cv.imread(os.path.join(self.mask_path,self.mask_names_list[index]),cv.IMREAD_GRAYSCALE) 
        #the unlabeld will be 182 rather than 255
        mask[mask==255]=182
        #print(mask.max())
        #mask grayscale 

        if self.transform:
            img=self.transform(img)
            mask=self.transform(mask)

        return (img,mask)

if __name__=="__main__":
    mytransform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480,640)),
            transforms.ConvertImageDtype(torch.float),
        ])
    dataset=Coco_Stuff_things(mytransform)
    data_loader=DataLoader(dataset=dataset,batch_size=2,shuffle=True)

    for (imgs,masks) in data_loader:
        #print(imgs.shape)
        #print(masks.shape)
        for ind in range(imgs.shape[0]):
            a=imgs[ind,0,:,:].squeeze().detach().to('cpu').numpy()
            plt.subplot(1,2,1)
            plt.title("img")
            plt.imshow(a)
            plt.subplot(1,2,2)
            plt.title("ground truth")
            plt.imshow(masks[ind,0,:,:].squeeze().detach().to('cpu').numpy())
            plt.show()
