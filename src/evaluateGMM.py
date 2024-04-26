# SUR project 2023/2024
# Diana Maxima Držíková (xdrzik01), Mária Novákova (xnovak2w)
# Module for evaluating the GMM

from ikrlib import mfcc, train_gauss, gellipse, logpdf_gauss, logpdf_gmm, train_gmm
import pickle
import numpy as np
from scipy.io import wavfile
import os
import argparse


def evaluate(input, output, target, non_target):
    """ Function for evaluating the GMM

    Args:
        input: Filepath for input files
        output: Filepath for output file (predictions are written here)
        target: Filepath for model for target
        non_target: Filepath for model non target
    """
    # load the models
    with open(target, 'rb') as file:  
        gmm_target_model = pickle.load(file)

    with open(non_target, 'rb') as file:
        gmm_nontarget_model = pickle.load(file)

    Ws_t = gmm_target_model['weights']
    MUs_t = gmm_target_model['means']
    COVs_t = gmm_target_model['covariances']

    Ws_nt = gmm_nontarget_model['weights']
    MUs_nt = gmm_nontarget_model['means']
    COVs_nt = gmm_nontarget_model['covariances']

    evaluated = []
    ones = 0
    zeros = 0
    # load audio files
    for root, _, files in os.walk(input):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)

                fs, data = wavfile.read(file_path)

                lenght = data.size / fs * 1000
                one_sample_cointains = lenght / data.size
                cut = int(2000.0/one_sample_cointains)

                # cut first two seconds
                data = data[cut:]
                data = data - np.mean(data) 
                data = data / 2**15 

                samples_for_one_frame = int(25.0/one_sample_cointains)
                samples_for_shift = int(10.0/one_sample_cointains)

                features = mfcc(data, samples_for_one_frame, samples_for_shift, 512, fs, 23, 13)

                # compute likehoods
                ll_t = logpdf_gmm(features, Ws_t, MUs_t, COVs_t)
                ll_nt = logpdf_gmm(features, Ws_nt, MUs_nt, COVs_nt)
                
                #s = sum(ll_m) - sum(ll_f) 
                s = sum(ll_t) * 0.5  - sum(ll_nt) * 0.5

                if int(s > 0):
                    ones += 1
                else:
                    zeros += 1

                evaluated.append({'soft':s, 'hard': int(s > 0), 'filename':file})


    with open(output, 'w') as f:
        for eval in evaluated:
            print(f"{eval['filename'][:-4]} {eval['soft']:.4f} {eval['hard']}", file=f)

    print(f"Target: {(ones/(ones+zeros)*100):.2f}%")
    print(f"Non-target: {(zeros/(ones+zeros)*100):.2f}%")


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
        default="../models/GMM/new/predictions_audio.txt",
        help='Output file path')
    
    parser.add_argument(
        '--target', 
        type=str, 
        default="../models/GMM/trained/gmm_target_model.pkl", # "../models/GMM/new/gmm_target_model.pkl"
        help='File with parameters of Gaussian for target')
    
    parser.add_argument(
        '--non_target', 
        type=str, 
        default="../models/GMM/trained/gmm_nontarget_model.pkl", #"../models/GMM/new/gmm_nontarget_model.pkl"
        help='File with parameters of Gaussian for non target')
    
    args = parser.parse_args()

    output = args.output
    target = args.target
    non_target = args.non_target
    input = args.input

    evaluate(input, output, target, non_target)
