import os
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser

def create_pearson_labels(dataname, folderpath, num_classes):   
    if not os.path.exists(folderpath + '/labels'): os.mkdir(folderpath + '/labels')

    segmentation_classes = list(range(0,num_classes))
    name_of_labels = (folderpath + 'masks/' + dataname + '_mask.npy').replace('_all_classes', '')
    mask_orig = np.load(name_of_labels)

    p = ImzMLParser(folderpath + dataname + '.imzML')
    all_mz, _ = p.getspectrum(0)

    all_spectra = []
    XCoord = []
    YCoord = []
    for idx, (x,y,z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        all_spectra.append(intensities)
        XCoord.append(x)
        YCoord.append(y)

    for class_number in segmentation_classes:
        mask = np.where(mask_orig == class_number, 1, 0)

        all_spectra = np.array(all_spectra)
        mask_flattened = []
        for i in range(len(XCoord)):
            mask_flattened.append(mask[YCoord[i]-1, XCoord[i]-1])

        pearson_correlations = []
        for idx, mz in enumerate(all_mz):
            ion_image = all_spectra[:,idx]
            pearson_corr = stats.pearsonr(mask_flattened, ion_image).correlation

            if math.isnan(pearson_corr):
                pearson_correlations.append(0)
            else:
                pearson_correlations.append(pearson_corr)

        ranking = np.argsort(pearson_correlations)[::-1]
        mz_values = [all_mz[rank] for rank in ranking]
        pearson_correlations = [pearson_correlations[rank] for rank in ranking]

        np.save(folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_ranking.npy', ranking)
        np.save(folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_mz_ranking.npy', mz_values)
        np.save(folderpath + 'labels/' + dataname + '_class' + str(class_number) + '_pearson_ranking.npy', pearson_correlations)