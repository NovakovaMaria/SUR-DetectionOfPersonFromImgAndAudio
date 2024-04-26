# SUR project 2023/2024
# Diana Maxima Držíková (xdrzik01), Mária Novákova (xnovak2w)
# Module for training the GMM

import os
from scipy.io import wavfile
import librosa, librosa.display
import scipy
import numpy as np
import matplotlib.pyplot as plt
from libGMM import wavToMel, plotROCAUC, GMM
import sys
from ikrlib import mfcc, train_gauss, gellipse, logpdf_gauss, logpdf_gmm, train_gmm
import argparse

def train(cv):
    """ Function for training the GMM

    Args:
        cv: Argument condition for allowing Cross-validaton
    """

    # load Dataset
    non_target_test, fs = wavToMel("../SUR_projekt2023-2024/non_target_dev")
    non_target_train, _ = wavToMel("../SUR_projekt2023-2024/non_target_train")

    target_test, _ = wavToMel("../SUR_projekt2023-2024/target_dev")
    target_train, _ = wavToMel("../SUR_projekt2023-2024/target_train")

    non_target_train = np.vstack(non_target_train)
    target_train = np.vstack(target_train)

    dim = non_target_train.shape[1]

    n_nt = len(non_target_train)
    n_t = len(target_train)

    # PCA

    data_combined = np.vstack([non_target_train, target_train])

    cov_tot = np.cov(data_combined.T, bias=True)

    d, e = scipy.linalg.eigh(cov_tot, subset_by_index=(dim-2, dim-1))

    # LDA

    cov_wc = (n_nt*np.cov(non_target_train.T, bias=True) + n_t*np.cov(target_train.T, bias=True)) / (n_nt + n_t)
    cov_ac = cov_tot - cov_wc

    d, e = scipy.linalg.eigh(cov_ac, cov_wc, subset_by_index=(dim-1, dim-1))

    plt.figure()
    junk = plt.hist(target_train.dot(e), 40, histtype='step', color='b', density=True, label="Target")
    junk = plt.hist(non_target_train.dot(e), 40, histtype='step', color='r', density=True, label="Non Target")
    plt.legend(loc="upper right")
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('LDA')
    plt.savefig("../models/GMM/new/LDA_GMM.png")

    # GMM
    combined_scores = GMM(target_train, target_test, non_target_train, non_target_test, n_t, n_nt, cv)

    true_labels_target = np.ones(len(target_test))
    true_labels_non_target = np.zeros(len(non_target_test))
    true_labels = np.concatenate((true_labels_target, true_labels_non_target))

    plotROCAUC(true_labels, combined_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cv', action='store_true', default=False,
        help='Cross-validation will be exectured')
    
    args = parser.parse_args()

    cv = args.cv

    train(cv)
