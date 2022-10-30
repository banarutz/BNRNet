import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def generate_paths(dir_list):

    paths = []
    for dir in dir_list:
        for element in os.listdir(dir):
            full_path = os.path.join(dir, element)
            paths.append(full_path)
    return paths

class VGGLoader (torch.utils.data.Dataset):
    def __init__ (self, PATH):
        self.PATH = PATH
        self.paths = os.listdir(PATH)

    def __getitem__ (self, index):
        path = self.paths[index]

        image = cv2.imread(self.PATH + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

############  ADD CHECKS FOR DIMENSIONS #################
#         transforms = torch.nn.Sequential(
#     transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), ### Oare e bine sa fac normalizare si dupa sa transform in LAB?
# )

        image = torch.Tensor(image)
        input = image [0, :, :]

        return input, image
        

    def __len__ (self):
        return len(self.paths)


class ValidationDataSet (torch.utils.data.Dataset):
    def __init__(self):

        pass





