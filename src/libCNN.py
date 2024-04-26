# SUR project 2023/2024
# Mária Novákova (xnovak2w), Diana Maxima Držíková (xdrzik01)
# Library with functions for training CNN

import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class AddNoise:
    def __init__(self, mean=0., std=0.1):
        """Add random noise to an image tensor.
            Args:
                mean: Mean of the noise
                std: Standard deviation
        """
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MergeDataset(Dataset):
    def __init__(self, images, labels, transform=None, augmentations=5):
        """ Class for creating dataset with transformations (for training and testing dataset)
            Args:
                images: Array of image data or paths to images.
                labels: Array of labels.
                transform: Optional transform to be applied on an image.
        """
        self.images = images
        self.labels = labels.astype(int)
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        original_idx = idx // self.augmentations
        image = Image.fromarray(self.images[original_idx])
        label = self.labels[original_idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
    
class EvalDataset(Dataset):
    def __init__(self, image_files, transform=None):
        """ Class for creating dataset with transformations (for evaluation) returning image and its name
            Args:
                image_files: Each tuple contains an image (PIL Image) and its path.
                transform: Optional transform to be applied on a PIL image.
        """
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image, path = self.image_files[idx]
        if self.transform:
            image = self.transform(image)
        return image, path


def loadImg(directory, type=True):
    """ Function for loading the input images

    Args:
        directory: Filepath to directory containing input images
        type: Condition determinig whether the call is from evaluation or training module

    Returns:
        array: Array of loaded images
    """
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                image = Image.open(file_path).convert('RGB')
                if type:
                    image = np.array(image)
                    image_files.append(image)
                else:
                    image_files.append((image, file))
    return image_files


class CNN(nn.Module):
    def __init__(self):
        """ Architecture of Convolutional Neural Network
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 1)  
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.apply(self.init_weights)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 64 * 10 * 10)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)



def plotROCAUC(score_true, score_predicted):
    """ Function for plotting ROC/AUC curve

    Args:
        score_true: Groung Truths
        score_predicted: Output probabilites from model
    """

    fpr, tpr, thresholds = roc_curve(score_true, score_predicted)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("../models/CNN/new/rocauc.png")
