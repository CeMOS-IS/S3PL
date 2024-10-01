import os
import numpy as np

from utils.create_pearson_labels import create_pearson_labels

def tic_norm_spectra(arr):
    max_vals = np.max(arr, axis=2, keepdims=True)
    max_vals[max_vals == 0] = 1 # avoid division by zero
    normalized = arr / max_vals
    return normalized

def check_for_labels(folderpath, dataname):
    if not os.path.exists(folderpath + 'masks/' + dataname + '_mask.npy'):
        raise Exception('There is no mask given for the file ' + dataname + '.imzML in the folder "' + folderpath + 'masks/". The mask should have the name "' + dataname + '_mask.npy".')
    
    num_classes = len(np.unique(np.load(folderpath + 'masks/' + dataname + '_mask.npy')))
    labels_are_missing = False
    for class_number in range(num_classes):
        label_ranking = folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_ranking.npy'
        label_mz_ranking = folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_mz_ranking.npy'
        label_pearson_ranking = folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_pearson_ranking.npy'

        if not os.path.exists(label_ranking):
            labels_are_missing = True
        if not os.path.exists(label_mz_ranking):
            labels_are_missing = True
        if not os.path.exists(label_pearson_ranking):
            labels_are_missing = True
    
    if labels_are_missing:
        create_pearson_labels(dataname, folderpath, num_classes)