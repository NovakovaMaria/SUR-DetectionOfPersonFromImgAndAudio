# SUR project 2023/2024
# Mária Novákova (xnovak2w), Diana Maxima Držíková (xdrzik01)
# Module for evaluating the CNN

import torch
import numpy as np
from torch import nn
from libCNN import loadImg, CNN, MergeDataset, EvalDataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse


def evaluate(input, output, threshold, model_path):
    """ Function for evaluating the CNN

    Args:
        input: Filepath for input files
        output: Filepath for output file (predictions are written here)
        model_path: Filepath with parameters of the trained model
    """

    # load model
    model = CNN() 
    model.load_state_dict(torch.load(model_path))
    model.eval()

    evaluate_data = loadImg(input, False)

    # define transformations for images
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create validation set
    validation_dataset = EvalDataset(evaluate_data,  transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    predictions = []

    # evaluate
    with torch.no_grad():
        for images, paths in validation_loader:
            logits = model(images)
            probs = torch.sigmoid(logits)
            predicted = (probs > threshold).long()

            for path, prob, pred in zip(paths, probs, predicted):
                predictions.append((path, prob.item(), pred.item()))

    # write to the file
    with open(output, 'w') as f:
        for path, prob, pred in predictions:
            print(f"{path[:-4]} {prob:.4f} {pred}", file=f)

    # count percentages
    class_counts = [0, 0] 
    for _, _, pred in predictions:
        class_counts[pred] += 1

    total_predictions = len(predictions)
    if total_predictions > 0:
        print(f"Target: {(class_counts[1]/total_predictions*100):.2f}%")
        print(f"Non-target: {(class_counts[0]/total_predictions*100):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', 
        type=str, 
        default="../SUR_projekt2023-2024_eval",
        help='Input files path')
    
    parser.add_argument(
        '--output', 
        type=str, 
        default="../models/CNN/new/predictions_img.txt",
        help='Output file path')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default="../models/CNN/new/model.pth",
        help='File with CNN model')
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Threshold for hard decision')

    args = parser.parse_args()

    threshold = args.threshold
    output = args.output
    input = args.input
    model = args.model

    evaluate(input, output, threshold, model)
