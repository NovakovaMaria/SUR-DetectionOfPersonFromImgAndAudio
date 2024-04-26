# SUR project 2023/2024
# Mária Novákova (xnovak2w), Diana Maxima Držíková (xdrzik01)
# Module for training the CNN

import torch
import numpy as np
from torch import nn
from libCNN import loadImg, CNN, MergeDataset, AddNoise, plotROCAUC
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

def train(threshold):
    """ Function for training the CNN model

    Args:
        threshold: Threshold for hard decision
    """
        
    non_target_test = loadImg("../SUR_projekt2023-2024/non_target_dev")
    non_target_train = loadImg("../SUR_projekt2023-2024/non_target_train")
    target_test = loadImg("../SUR_projekt2023-2024/target_dev")
    target_train = loadImg("../SUR_projekt2023-2024/target_train")

    # create labels for supervised traning
    target_labels = np.ones(len(target_train))
    non_target_labels = np.zeros(len(non_target_train))
    target_test_labels = np.ones(len(target_test))
    non_target_test_labels = np.zeros(len(non_target_test))

    # define transformations for images
    transform = transforms.Compose([
        transforms.Resize((80, 80)),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(20),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
        transforms.RandomResizedCrop(80, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        AddNoise(mean=0., std=0.06),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    images = np.concatenate((target_train, non_target_train), axis=0)
    labels = np.concatenate((target_labels, non_target_labels), axis=0)

    # create training dataset
    train_dataset = MergeDataset(images, labels, transform=transform, augmentations=5)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CNN()

    loss = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=loss)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.float().unsqueeze(1)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step() 

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

    transformations = transforms.Compose([
        transforms.Resize((80, 80)),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    images = np.concatenate((target_test, non_target_test), axis=0)
    labels = np.concatenate((target_test_labels, non_target_test_labels), axis=0)

    # create testing dataset
    test_dataset = MergeDataset(images, labels, transform=transformations)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model.eval() 
    total = 0
    correct = 0
    all_labels = []
    all_probs = []

    # test the model
    with torch.no_grad():  
        for images, labels in test_loader:
            logits = model(images)
            
            probs = torch.sigmoid(logits)[:, 0] 
            
            predicted = (probs > threshold).long() 
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), "../models/CNN/new/model.pth")

    plotROCAUC(all_labels, all_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Threshold for hard decision')
    
    args = parser.parse_args()
    threshold = args.threshold

    train(threshold)
