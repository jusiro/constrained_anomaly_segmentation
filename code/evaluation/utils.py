import sklearn.metrics
import imutils
import cv2
import os
import matplotlib.pyplot as plt

from evaluation.metrics import *


def inference_dataset(method, dataset):

    # Take references and inputs from testing dataset
    X = dataset.X
    M = dataset.M
    Y = dataset.Y

    if 'BRATS' in dataset.dir_datasets or 'PhysioNet' in dataset.dir_datasets:  # Inference is volume-wise

        if len(X.shape) > 5:
            X = X[:, 0, :, :, :, :]

        if len(M.shape) < 5:
            M = np.expand_dims(M, 1)

        # Init variables
        (p, s, c, h, w) = X.shape  # maps dimensions
        Mhat = np.zeros(M.shape)  # Predicted segmentation maps
        Xhat = np.zeros(X.shape)  # Reconstructed images
        Scores = np.zeros((p, s))

        for iVolume in np.arange(0, p):
            for iSlice in np.arange(0, s):
                # Take image
                x = X[iVolume, iSlice, :, :, :]
                # Predict anomaly map and score
                score, mhat, xhat = method.predict_score(x)

                Mhat[iVolume, iSlice, :, :, :] = mhat
                Xhat[iVolume, iSlice, :, :, :] = xhat
                Scores[iVolume, iSlice] = score

    elif 'MVTEC' in dataset.dir_datasets:  # Inference is image-wise

        # Init variables
        (cases, c, h, w) = X.shape  # maps dimensions
        Mhat = np.zeros(M.shape)  # Predicted segmentation maps
        Xhat = np.zeros(X.shape)  # Reconstructed images
        Scores = np.zeros((cases, 1))

        for iCase in np.arange(0, cases):
            # Take image
            x = X[iCase, :, :, :]
            # Predict anomaly map and score
            score, mhat, xhat = method.predict_score(x)

            Mhat[iCase, :, :, :] = mhat
            Xhat[iCase, :, :, :] = xhat
            Scores[iCase, :] = score

    return Y, Scores, M, Mhat, X, Xhat


def evaluate_anomaly_detection(y, scores, dir_out='', range=[-1, 1], tit='cosine similarity', bins=50, th=None):

    scores = np.ravel(scores)
    y = np.ravel(y)

    auroc = sklearn.metrics.roc_auc_score(y, scores)  # au_roc
    auprc, th_op = au_prc(y, scores)  # au_prc

    if th is None:
        th = th_op

    if dir_out != '':
        plt.hist(np.ravel(scores)[np.ravel(y) == 1], bins=bins, range=range, fc=[0.7, 0, 0, 0.5])
        plt.hist(np.ravel(scores)[np.ravel(y) == 0], bins=bins, range=range, fc=[0, 0, 0.7, 0.5])
        plt.legend(['Anomaly', 'Normal'])
        plt.xlabel(tit)

        plt.savefig(dir_out + 'anomaly_detection.png')
        plt.close('all')

    return auroc, auprc, th


def evaluate_anomaly_localization(dataset, save_maps=False, dir_out='', filter_volumes=True, th=None):

    print('[INFO]: Testing...')
    if save_maps:
        if not os.path.isdir(dir_out + 'masks_predicted/'):
            os.mkdir(dir_out + 'masks_predicted/')
        if not os.path.isdir(dir_out + 'masks_reference'):
            os.mkdir(dir_out + 'masks_reference/')
        if not os.path.isdir(dir_out + 'xhat_predicted'):
            os.mkdir(dir_out + 'xhat_predicted/')

    # Get references and predictions from dataset
    M = dataset.M
    Mhat = dataset.Mhat
    X = dataset.X
    Xhat = dataset.Xhat

    if 'BRATS' in dataset.dir_datasets or 'PhysioNet' in dataset.dir_datasets:  # Inference is volume-wise

        if len(X.shape) > 5:
            X = X[:, 0, :, :, :, :]

        if filter_volumes:  # Filver volumes withouth annotations
            idx = np.squeeze(np.argwhere(np.sum(M, (1, 2, 3, 4)) / (M.shape[1] * M.shape[-1] * M.shape[-1]) > 0.001))
            X = X[idx, :, :, :, :]
            Xhat = Xhat[idx, :, :, :, :]
            M = M[idx, :, :, :, :]
            Mhat = Mhat[idx, :, :, :, :]
            unique_patients = list(np.array(dataset.unique_patients)[idx])

    # Obtain overall metrics and optimum point threshold
    AU_ROC = sklearn.metrics.roc_auc_score(M.flatten() == 1, Mhat.flatten())  # au_roc
    AU_PRC, th_op = au_prc(M.flatten() == 1, Mhat.flatten())  # au_prc

    if th is None:
        th = th_op

    # Apply threshold
    DICE = dice(M.flatten() == 1, (Mhat > th).flatten())  # Dice
    IoU = sklearn.metrics.jaccard_score(M.flatten() == 1, (Mhat > th).flatten())  # IoU

    if 'BRATS' in dataset.dir_datasets or 'PhysioNet' in dataset.dir_datasets:  # Inference is volume-wise

        # Once the threshold is obtained calculate volume-level metrics and plot results
        patient_dice = []
        (p, s, c, h, w) = X.shape  # maps dimensions
        for iVolume in np.arange(0, p):
            patient_dice.append(dice(M[iVolume, :, :, :, :].flatten(), (Mhat[iVolume, :, :, :, :] > th).flatten()))

            if save_maps and dir_out!= '':  # Save slices' masks
                for iSlice in np.arange(0, s):
                    id = unique_patients[iVolume] + '_' + str(iSlice) + '.jpg'

                    # Obtain heatmaps for predicted and reference
                    m_i = imutils.rotate_bound(np.uint8(M[iVolume, iSlice, 0, :, :] * 255), 270)
                    heatmap_m = cv2.applyColorMap(m_i, cv2.COLORMAP_JET)
                    mhat_i = imutils.rotate_bound(np.uint8((Mhat[iVolume, iSlice, 0, :, :] > th) * 255), 270)
                    heatmap_mhat = cv2.applyColorMap(mhat_i, cv2.COLORMAP_JET)
                    heatmap_mhat = heatmap_mhat * (
                                np.expand_dims(imutils.rotate_bound(Mhat[iVolume, iSlice, 0, :, :], 270), -1) > 0)

                    # Move grayscale image to three channels
                    xh = cv2.cvtColor(np.uint8(np.squeeze(X[iVolume, iSlice, :, :, :]) * 255), cv2.COLOR_GRAY2RGB)
                    xh = imutils.rotate_bound(xh, 270)

                    # Combine original image and masks
                    fin_mask = cv2.addWeighted(xh, 0.7, heatmap_m, 0.3, 0)
                    fin_predicted = cv2.addWeighted(xh, 0.7, heatmap_mhat, 0.3, 0)

                    fin_predicted = mhat_i
                    fin_mask = m_i

                    cv2.imwrite(dir_out + 'masks_predicted/' + id, fin_predicted)
                    cv2.imwrite(dir_out + 'masks_reference/' + id, fin_mask)
                    cv2.imwrite(dir_out + 'xhat_predicted/' + id, np.uint8(Xhat[iVolume, iSlice, 0, :, :] * 255))

        DICE_mu = np.mean(patient_dice)
        DICE_std = np.std(patient_dice)

    if 'MVTEC' in dataset.dir_datasets:  # Inference is image-wise

        # Once the threshold is obtained calculate volume-level metrics and plot results
        case_dice = []
        (cases, c, h, w) = X.shape  # maps dimensions
        for iCase in np.arange(0, cases):
            if 'good' not in dataset.images[iCase]:
                case_dice.append(dice(M[iCase, :, :, :].flatten(), (Mhat[iCase, :, :, :] > th).flatten()))

            if save_maps and dir_out != '':  # Save slices' masks
                id = dataset.images[iCase].replace('.png', '.jpg').split('/')[-2] + '_' + \
                     dataset.images[iCase].replace('.png', '.jpg').split('/')[-1]

                # Obtain heatmaps for predicted and reference
                m_i = np.uint8(M[iCase, 0, :, :] * 255)
                heatmap_m = cv2.applyColorMap(m_i, cv2.COLORMAP_JET)
                mhat_i = np.uint8((Mhat[iCase, 0, :, :] > th) * 255)
                heatmap_mhat = cv2.applyColorMap(mhat_i, cv2.COLORMAP_JET)

                # Move grayscale image to three channels
                xh = np.uint8(np.squeeze(X[iCase, :, :, :]) * 255)
                xh = np.transpose(xh, (1, 2, 0))

                # Combine original image and masks
                fin_mask = cv2.addWeighted(xh, 0.7, heatmap_m, 0.3, 0)
                fin_predicted = cv2.addWeighted(xh, 0.7, heatmap_mhat, 0.3, 0)

                cv2.imwrite(dir_out + 'masks_predicted/' + id, fin_predicted)
                cv2.imwrite(dir_out + 'masks_reference/' + id, fin_mask)
                cv2.imwrite(dir_out + 'xhat_predicted/' + id, np.uint8(Xhat[iCase, 0, :, :] * 255))

        DICE_mu = np.mean(case_dice)
        DICE_std = np.std(case_dice)

    metrics = {'AU_ROC': AU_ROC, 'AU_PRC': AU_PRC, 'DICE': DICE, 'IoU': IoU,
               'DICE_mu': DICE_mu, 'DICE_std': DICE_std}

    return metrics, th