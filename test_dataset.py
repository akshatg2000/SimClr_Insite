import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import time
import torchvision
import torch.nn as nn
from torchvision.io import read_image
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os


class InsiteDatasetTrain(Dataset):

    def __init__(self, npz_dir,transform):

        self.npz = np.load(npz_dir)
        self.img=self.npz['train_images']
        self.labels=self.npz['train_labels']
        self.transform=transform
    def __len__(self):
    
        return len(self.labels)

    def __getitem__(self, idx):
        cur_img=self.img[idx,:,:,:]
        cur_img=(cur_img*255).astype('uint8')
        # print(cur_img.shape)
        image = Image.fromarray(cur_img)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image,label
    

class InsiteDatasetTrainSubset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    

class InsiteDatasetTestSubset(Dataset):
    def __init__(self, dataset, selected_indices):
        self.dataset = dataset
        self.selected_indices = selected_indices

    def __len__(self):
        # The length of the CustomSubset is the number of samples not in selected_indices
        return len(self.dataset) - len(self.selected_indices)

    def __getitem__(self, idx):
        # Calculate the index for the dataset sample not in selected_indices
        non_selected_indices = [i for i in range(len(self.dataset)) if i not in self.selected_indices]
        return self.dataset[non_selected_indices[idx]]

    
