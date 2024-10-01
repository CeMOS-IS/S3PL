import numpy as np
import matplotlib.pyplot as plt

from pyimzml.ImzMLParser import ImzMLParser, getionimage

class PeakEvaluation:
    """
    Evaluation of a peak picking method based on a segmentation mask. Ion images with a higher Pearson correlation coefficient (PCC) to any region in the segmentation mask than `pearson_threshold` are labeled as true peaks.

    This class expects the name of the imzML file, the number of classes, the PCC threshold defining positive and negative peaks, the folderpath to the data, the peak list containing the picked m/z values.
    Optional: show picked ion images, normalize results in case of major deviations between the number of ground truth positive peaks and the number of picked peaks.
    """

    def __init__(self, dataname, num_classes, pearson_threshold, folderpath, peak_list, show_ion_images = False, normalise_results = False):
        self.dataname = dataname
        self.num_classes = num_classes
        self.pearson_threshold = pearson_threshold
        self.peak_list = peak_list
        self.folderpath = folderpath
        self.show_ion_images = show_ion_images
        self.number_picked_peaks = len(peak_list)
        self.normalise_results = normalise_results

        self.parser = ImzMLParser(folderpath + dataname + '.imzML')
        self.all_mz, _ = self.parser.getspectrum(0)

        for idx, mz in enumerate(self.peak_list):
            if mz not in self.all_mz:
                adapted_index = self.get_closest_index(self.all_mz, mz)
                self.peak_list[idx] = self.all_mz[adapted_index]
                
        self.non_peaks = []
        for mz in self.all_mz:
            if mz not in peak_list:
                self.non_peaks.append(mz)

    def get_true_false(self, values, num_best):
        """Expects a list of sorted values and returns the first "num_best" values of that list as true values and the rest as false values"""

        gt_true = values[:num_best]
        gt_false = values[num_best:]
        return gt_true, gt_false

    def get_groundtruth(self):
        """Returns two list of indices corresponding to the true peaks and the false peaks"""

        self.true_peaks = []
        for class_id in self.num_classes:
            idx_ranking = np.load(self.folderpath + 'labels/' + self.dataname + '_class' + str(class_id) + '_ranking.npy')
            pearson_ranking = np.load(self.folderpath + 'labels/' + self.dataname + '_class' + str(class_id) + '_pearson_ranking.npy')
            num_best_peaks = np.argmin(abs(pearson_ranking - self.pearson_threshold))
            gt_true = self.get_true_false(idx_ranking, num_best_peaks)[0]
            self.true_peaks.extend(gt_true)

        self.false_peaks = []
        for mz in idx_ranking:
            if mz not in self.true_peaks:
                self.false_peaks.append(mz)

        self.true_peaks = [self.all_mz[idx] for idx in self.true_peaks]
        self.false_peaks = [self.all_mz[idx] for idx in self.false_peaks]
        return self.true_peaks, self.false_peaks
    
    def get_closest_index(self, value_list, value):
        abs_difference = np.abs(np.array(value_list) - value)
        return(np.argmin(abs_difference))

    def calculate_metrics(self):
        """Calculates recall, precision and F1-score using the selected and not selected peaks as well as true peaks and false peaks"""

        self.true_peaks, self.false_peaks = self.get_groundtruth()

        TP = len(set(self.peak_list) & set(self.true_peaks))
        FP = len(set(self.peak_list) & set(self.false_peaks))
        FN = len(set(self.non_peaks) & set(self.true_peaks)) 
        TN = len(set(self.non_peaks) & set(self.false_peaks))
        
        if self.show_ion_images:
            for peak in self.peak_list:
                img = getionimage(self.parser, peak, tol=0.0001)
                true_peak = peak in self.true_peaks
                plt.imshow(img)
                plt.title('m/z ' + str(peak) + ' part of wanted peaks: ' + str(true_peak))
                plt.show()

        if TP+FP+TN+FN != len(self.all_mz):
            raise ValueError('The m/z values are not in their original format (some m/z values are neither part of true_peaks nor part of false_peaks). Probabably they got rounded')
        
        print('There are ' + str(len(self.true_peaks)) + ' true peaks of ' + str(len(self.all_mz)) + ' peaks in total')
        if TP == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        F1 = 2*TP/(2*TP + FP + FN)

        if self.normalise_results:
            if self.number_picked_peaks < len(self.true_peaks):
                TP_max = self.number_picked_peaks
                FN_min = len(self.true_peaks) - self.number_picked_peaks
                max_recall = TP_max/(TP_max+FN_min)
                max_f1 = 2*(1*max_recall)/(1+max_recall)

                recall = recall / max_recall
                F1 = F1 / max_f1
                print('Recall and F1-Score have been normalised to the max possible values of ' + str(round(max_recall,2)) + ' and ' + str(round(max_f1,2)) + ' for recall and f1.')

            if self.number_picked_peaks > len(self.true_peaks):
                TP_max = len(self.true_peaks)
                FP_min = self.number_picked_peaks - len(self.true_peaks)
                max_precision =  TP_max/(TP_max+FP_min)
                max_f1 = 2*(max_precision*1)/(max_precision+1)

                precision = precision / max_precision
                F1 = F1 / max_f1
                print('Precision and F1-Score have been normalised to the max possible values of ' + str(round(max_precision,2)) + ' and ' + str(round(max_f1,2)) + ' for precision and f1.')

        return recall, precision, F1
    
class PeakEvaluationMultipleClasses:
    """
    Evaluation of a peak picking method based on a segmentation mask. This class extends the PeakEvaluation class by providing "class-wise" evaluation of peak picking. The "class-wise" metrics are calculated by incorporating only the true peaks from class x and not all classes. The peak lists are the same, as peak picking methods do not perform classification of peaks.
    Ion images with a higher Pearson correlation coefficient (PCC) to any region in the segmentation mask than `pearson_threshold` are labeled as true peaks.

    This class expects the name of the imzML file, the number of classes, the PCC threshold defining positive and negative peaks, the folderpath to the data, the peak list containing the picked m/z values.
    Optional: show picked ion images, normalize results in case of major deviations between the number of ground truth positive peaks and the number of picked peaks.
    """
        
    def __init__(self, dataname, num_classes, pearson_threshold, folderpath, peak_list, show_ion_images = False, normalise_results = False):
        self.dataname = dataname
        self.num_classes = num_classes
        self.pearson_threshold = pearson_threshold
        self.peak_list = peak_list
        self.folderpath = folderpath
        self.show_ion_images = show_ion_images
        self.number_picked_peaks = len(peak_list)
        self.normalise_results = normalise_results

        self.parser = ImzMLParser(folderpath + dataname + '.imzML')
        self.all_mz, _ = self.parser.getspectrum(0)

        for idx, mz in enumerate(self.peak_list):
            if mz not in self.all_mz:
                adapted_index = self.get_closest_index(self.all_mz, mz)
                self.peak_list[idx] = self.all_mz[adapted_index]

        self.non_peaks = []
        for mz in self.all_mz:
            if mz not in peak_list:
                self.non_peaks.append(mz)

    def get_true_false(self, values, num_best, datatype = 'mz'):
        """Expects a list of sorted values and returns the first "num_best" values of that list as true values and the rest as false values"""

        gt_true = values[:num_best]
        gt_false = values[num_best:]
        return gt_true, gt_false

    def get_groundtruth(self):
        """Returns two list of indices corresponding to the true peaks and the false peaks"""

        self.groundtruth = {}
        self.true_peaks = []
        for class_id in self.num_classes:
            idx_ranking = np.load(self.folderpath + 'labels/' + self.dataname + '_class' + str(class_id) + '_ranking.npy')
            pearson_ranking = np.load(self.folderpath + 'labels/' + self.dataname + '_class' + str(class_id) + '_pearson_ranking.npy')
            num_best_peaks = np.argmin(abs(pearson_ranking - self.pearson_threshold))
            gt_true = self.get_true_false(idx_ranking, num_best_peaks)[0]
            self.true_peaks.extend(gt_true)
            self.groundtruth["true_peaks_class" + str(class_id)] = gt_true

            false_peaks_class = []
            for mz in idx_ranking:
                if mz not in self.groundtruth["true_peaks_class" + str(class_id)]:
                    false_peaks_class.append(mz)
                    self.groundtruth["false_peaks_class" + str(class_id)] = false_peaks_class

            self.groundtruth["true_peaks_class" + str(class_id)] = [self.all_mz[idx] for idx in self.groundtruth["true_peaks_class" + str(class_id)]]
            self.groundtruth["false_peaks_class" + str(class_id)] = [self.all_mz[idx] for idx in self.groundtruth["false_peaks_class" + str(class_id)]]

        self.false_peaks = []
        for mz in idx_ranking:
            if mz not in self.true_peaks:
                self.false_peaks.append(mz)

        self.true_peaks = [self.all_mz[idx] for idx in self.true_peaks]
        self.false_peaks = [self.all_mz[idx] for idx in self.false_peaks]
        return self.true_peaks, self.false_peaks
    
    def get_closest_index(self, value_list, value):
        abs_difference = np.abs(np.array(value_list) - value)
        return(np.argmin(abs_difference))

    def calculate_metrics(self):
        """Calculates recall, precision and F1-score using the selected and not selected peaks as well as true peaks and false peaks"""

        self.true_peaks, self.false_peaks = self.get_groundtruth()
        
        class_metrics = {}
        for class_id in self.num_classes:
            TP = len(set(self.peak_list) & set(self.groundtruth["true_peaks_class" + str(class_id)]))
            FP = len(set(self.peak_list) & set(self.groundtruth["false_peaks_class" + str(class_id)]))
            FN = len(set(self.non_peaks) & set(self.groundtruth["true_peaks_class" + str(class_id)])) 
            TN = len(set(self.non_peaks) & set(self.groundtruth["false_peaks_class" + str(class_id)]))
            
            if self.show_ion_images:
                for peak in self.peak_list:
                    img = getionimage(self.parser, peak, tol=0.0001)
                    true_peak = peak in self.groundtruth["true_peaks_class" + str(class_id)]
                    plt.imshow(img)
                    plt.title('m/z ' + str(peak) + ' part of wanted peaks: ' + str(true_peak))
                    plt.show()

            if TP+FP+TN+FN != len(self.all_mz):
                raise ValueError('The m/z values are not in their original format (some m/z values are neither part of true_peaks nor part of false_peaks). Probabably they got rounded')
            
            print('There are ' + str(len(self.groundtruth["true_peaks_class" + str(class_id)])) + ' true peaks of ' + str(len(self.all_mz)) + ' peaks in class ' + str(class_id))
            
            if TP == 0:
                recall = 0
            else:
                recall = TP/(TP+FN)
            precision = TP/(TP+FP)
            F1 = 2*TP/(2*TP + FP + FN)

            if self.normalise_results:
                if self.number_picked_peaks < len(self.groundtruth["true_peaks_class" + str(class_id)]):
                    TP_max = self.number_picked_peaks
                    FN_min = len(self.groundtruth["true_peaks_class" + str(class_id)]) - self.number_picked_peaks
                    max_recall = TP_max/(TP_max+FN_min)
                    max_f1 = 2*(1*max_recall)/(1+max_recall)

                    recall = recall / max_recall
                    F1 = F1 / max_f1
                    print('Recall and F1-Score have been normalised to the max possible values of ' + str(round(max_recall,2)) + ' and ' + str(round(max_f1,2)) + ' for recall and f1.')

                if self.number_picked_peaks > len(self.groundtruth["true_peaks_class" + str(class_id)]):
                    TP_max = len(self.groundtruth["true_peaks_class" + str(class_id)])
                    FP_min = self.number_picked_peaks - len(self.groundtruth["true_peaks_class" + str(class_id)])
                    max_precision =  TP_max/(TP_max+FP_min)
                    max_f1 = 2*(max_precision*1)/(max_precision+1)

                    precision = precision / max_precision
                    F1 = F1 / max_f1
                    print('Precision and F1-Score have been normalised to the max possible values of ' + str(round(max_precision,2)) + ' and ' + str(round(max_f1,2)) + ' for precision and f1.')

            metrics = {
                "recall": recall,
                "precision": precision,
                "F1": F1,
            }

            class_metrics["class " + str(class_id)] = metrics
        return class_metrics