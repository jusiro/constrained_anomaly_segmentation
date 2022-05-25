import datetime
import kornia
import json
import torch

import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from timeit import default_timer as timer
from models.models import Encoder, Decoder
from evaluation.utils import *
from methods.losses.losses import kl_loss
from datasets.utils import augment_input_batch
from skimage.exposure import equalize_hist


class AnomalyDetectorHistEqualization:
    def __init__(self, dir_results, item=['flair']):

        # Init input variables
        self.dir_results = dir_results
        self.item = item
        self.train_generator = []
        self.dataset_test = []

    def train(self, train_generator, epochs, dataset_test):
        self.train_generator = train_generator
        self.dataset_test = dataset_test
        print('No training for Histogram matching method.', end='\n')

        # Make predictions
        Y, Scores, M, Mhat, X, Xhat = inference_dataset(self, self.dataset_test)

        # Input to dataset
        self.dataset_test.Scores = Scores
        self.dataset_test.Mhat = Mhat
        self.dataset_test.Xhat = Xhat

        # Evaluate
        metrics, th = evaluate_anomaly_localization(self.dataset_test, save_maps=True, dir_out=self.dir_results)
        self.metrics = metrics

        # Save metrics as dict
        with open(self.dir_results + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

    def predict_score(self, x):
        def equalize_img(img):
            """
            Perform histogram equalization on the given image.
            """
            # Create equalization mask
            mask = np.zeros_like(img)
            mask[img > 0] = 1

            # Equalize
            img = img*255
            img = equalize_hist(img.astype(np.int64), nbins=256, mask=mask)

            # Assure that background still is 0
            img *= mask
            img *= (1/255)

            return img

        # Prepare brain eroded mask
        if 'BRATS' in self.train_generator.dataset.dir_datasets or 'PhysioNet' in self.train_generator.dataset.dir_datasets:
            x_mask = 1 - (x == 0).astype(np.int)
            if 'BRATS' in self.train_generator.dataset.dir_datasets:
                x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 10, 10))).astype(x_mask.dtype)
            else:
                x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 3, 3))).astype(x_mask.dtype)
        elif 'MVTEC' in self.train_generator.dataset.dir_datasets:
            x_mask = np.zeros((1, x.shape[-1], x.shape[-1]))
            x_mask[:, 14:-14, 14:-14] = 1

        # Get reconstruction error map
        mhat = equalize_img(x)
        mhat = mhat[0, :, :]

        # Keep only brain region
        mhat[x_mask[0, :, :] == 0] = 0

        # Get outputs
        anomaly_map = mhat
        score = np.mean(anomaly_map)

        return score, anomaly_map, x


