from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import numpy as np


def dice(true_mask, pred_mask, non_seg_score=1.0):

    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum


def au_prc(true_mask, pred_mask):

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0

    th = threshold[np.argmax(f1)]

    return au_prc, th