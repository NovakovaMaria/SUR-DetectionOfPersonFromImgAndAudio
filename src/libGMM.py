# SUR project 2023/2024
# Diana Maxima Držíková (xdrzik01), Mária Novákova (xnovak2w)
# Library with functions for training GMM

import os
from scipy.io import wavfile
import librosa, librosa.display
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from numpy.random import randint
from scipy.special import expit
from ikrlib import mfcc, train_gauss, gellipse, logpdf_gauss, logpdf_gmm, train_gmm
import pickle
from sklearn.model_selection import KFold

def augument(data, sr):
    """ Function for augumenting the data

    Args:
        data: Loaded audio recording
        sr: Sampling rate

    Returns:
       array: Array of augumented audio recording
    """

    augmented_data = []

    # augument noise
    wav_n = data + 0.009*np.random.normal(0,1,len(data))
    augmented_data.append(wav_n)

    # augument shift
    wav_roll = np.roll(data,int(sr/10))
    augmented_data.append(wav_roll)

    # augument stretching
    factor = 0.12
    wav_time_stch = librosa.effects.time_stretch(data,rate=factor)
    augmented_data.append(wav_time_stch)

    # augument pitch shift
    wav_pitch_sf = librosa.effects.pitch_shift(data,sr,n_steps=-5)
    augmented_data.append(wav_pitch_sf)
    
    return augmented_data

def wavToMel(directory):
    """ Loading and converting input audio recordings

    Args:
        directory: Filepath for source audio files

    Returns:
        array: Array with MFCCs of each audio (augumented)
        int: Sampling rate
    """

    audio_files = []
    for root, _, files in os.walk(directory):
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

                # augument the audio recordings
                aug_data = augument(data, fs)
                
                for ad in aug_data:
                    featuresaug = mfcc(ad, samples_for_one_frame, samples_for_shift, 512, fs, 23, 13)
                    audio_files.append(featuresaug)
                
                features = mfcc(data, samples_for_one_frame, samples_for_shift, 512, fs, 23, 13)
                audio_files.append(features)

    return audio_files, fs


def GMM(target_train, target_test, non_target_train, non_target_test, n_t, n_nt, cross_validation):
    """ Training the GMM

    Args:
        target_train: MFCCs for target training
        target_test: MFCCs for target testing
        non_target_train: MFCCs for non target training 
        non_target_test: MFCCs for non target testing
        n_t: Lenght of target training array
        n_nt: Lenght of non target training array
        cross_validation: Argument condition for allowing Cross-validaton

    Returns:
        array: Scores of testing data
    """

    P_t = n_t / (n_t+n_nt)
    P_nt = 1 - P_t  

    # Cross-validation
    if(cross_validation):
        data = np.concatenate((target_train, non_target_train))
        labels = np.array([1] * len(target_train) + [0] * len(non_target_train))
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        i = 3
        for train_index, test_index in kf.split(data):
            # Split data into training and validation sets
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Initialize GMM parameters for target class (class 1)
            target_indices = y_train == 1
            M_t = i
            MUs_t = X_train[target_indices][np.random.randint(0, sum(target_indices), M_t)]
            COVs_t = [np.var(X_train[target_indices], axis=0)] * M_t
            Ws_t = np.ones(M_t) / M_t

            # Initialize GMM parameters for non-target class (class 0)
            non_target_indices = y_train == 0
            M_nt = i
            MUs_nt = X_train[non_target_indices][np.random.randint(0, sum(non_target_indices), M_nt)]
            COVs_nt = [np.var(X_train[non_target_indices], axis=0)] * M_nt
            Ws_nt = np.ones(M_nt) / M_nt

            i += 1
            # Train GMMs for both classes
            for jj in range(10):
                Ws_t, MUs_t, COVs_t, TTL_t = train_gmm(X_train[target_indices], Ws_t, MUs_t, COVs_t)
                Ws_nt, MUs_nt, COVs_nt, TTL_nt = train_gmm(X_train[non_target_indices], Ws_nt, MUs_nt, COVs_nt)

                print('Iteration:', jj, ' Total log-likelihood:', TTL_t, 'for target;', TTL_nt, 'for non target')

            evaluate_gmm(X_test, y_test, Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt, P_t, P_nt)


    # Init GMM parameteres
    M_t = 5
    MUs_t  = target_train[randint(1, len(target_train), M_t)]
    COVs_t = [np.var(target_train, axis=0)] * M_t
    Ws_t   = np.ones(M_t) / M_t

    M_nt = 5
    MUs_nt  = non_target_train[randint(1, len(non_target_train), M_nt)]
    COVs_nt = [np.var(non_target_train, axis=0)] * M_nt
    Ws_nt   = np.ones(M_nt) / M_nt

    # Run 10 iterations of EM algorithm to train the two GMMs
    for jj in range(10):
        [Ws_t, MUs_t, COVs_t, TTL_t] = train_gmm(target_train, Ws_t, MUs_t, COVs_t); 
        [Ws_nt, MUs_nt, COVs_nt, TTL_nt] = train_gmm(non_target_train, Ws_nt, MUs_nt, COVs_nt); 
        print('Iteration:', jj, ' Total log-likelihood:', TTL_t, 'for target;', TTL_nt, 'for non target')

    # Evaluate on test dataset
    score_target = []
    score=[]
    # target evaluation
    for tst in target_test:
        ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
        s = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        score.append(s)
        score_target.append(s > 0)

    a = [x for x in score if x > 0.0]
    print(f"Target: {(len(a)/len(score)*100):.2f}%")

    score_for_target_test = expit(score)

    score_non_target = []
    score=[]
    # non target evaluation
    for tst in non_target_test:
        ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
        s = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        score.append(s)
        score_non_target.append(s < 0) 

    a = [x for x in score if x < 0.0]
    print(f"Non Target: {(len(a)/len(score)*100):.2f}%")

    all_results = score_target + score_non_target
    overall_accuracy = sum(all_results) / len(all_results)
    print(f"Overall accuracy: {(overall_accuracy*100):.2f}%")

    score_for_non_target_test = expit(score)

    combined_scores = np.concatenate((score_for_target_test, score_for_non_target_test))

    # save models
    with open('../models/GMM/new/gmm_target_model.pkl', 'wb') as file:  
        pickle.dump({'weights': Ws_t, 'means': MUs_t, 'covariances': COVs_t}, file)

    with open('../models/GMM/new/gmm_nontarget_model.pkl', 'wb') as file:
        pickle.dump({'weights': Ws_nt, 'means': MUs_nt, 'covariances': COVs_nt}, file)

    return combined_scores


def evaluate_gmm(X_test, y_test, Ws_t, MUs_t, COVs_t, Ws_nt, MUs_nt, COVs_nt, P_t, P_nt):
    """ Function for evaluating the model during cross validation

    Args:
        X_test: Target features
        y_test: Non target features
        Ws_t: Weights of target Gaussian
        MUs_t: Means of target Gaussian
        COVs_t: Covariance matrices of target Gaussian
        Ws_nt: Weight of non target Gaussian
        MUs_nt: Means of non target Gaussian
        COVs_nt: Covariance matrices of non target Gaussian
        P_t: Aprior probability of target
        P_nt: Aprior probability of non target
    """
    score=[]
    for tst, label in zip(X_test, y_test):
        ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
        score_value = (sum(ll_t) + np.log(P_t)) - (sum(ll_nt) + np.log(P_nt))
        score.append(score_value > 0 if label == 1 else score_value < 0)
    accuracy = sum(score) / len(score)
    print(f"Evaluation accuracy: {(accuracy*100):.2f}%")

def plotROCAUC(score_true, score_predicted):
    """ Function for plotting ROC/AUC curve

    Args:
        score_true: Groung Truths
        score_predicted: Output probabilites from model
    """

    fpr, tpr, _ = roc_curve(score_true, score_predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("../models/GMM/new/rocauc.png")
