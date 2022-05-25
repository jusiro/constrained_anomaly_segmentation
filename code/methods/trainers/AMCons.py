import torch
import numpy as np

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
from methods.losses.losses import log_barrier
from sklearn.metrics import accuracy_score, f1_score


class AnomalyDetectorAMCons:
    def __init__(self, dir_results, item=['flair'], zdim=32, lr=1*1e-4, input_shape=(1, 224, 224), epochs_to_test=25,
                 load_weigths=False, n_blocks=5, dense=True, loss_reconstruction='bce', alpha_kl=1,
                 pre_training_epochs=0, level_cams=-4, alpha_entropy=1, gap=False):

        # Init input variables
        self.dir_results = dir_results
        self.item = item
        self.zdim = zdim
        self.lr = lr
        self.input_shape = input_shape
        self.epochs_to_test = epochs_to_test
        self.load_weigths = load_weigths
        self.n_blocks = n_blocks
        self.dense = dense
        self.loss_reconstruction = loss_reconstruction
        self.alpha_kl = alpha_kl
        self.pre_training_epochs = pre_training_epochs
        self.level_cams = level_cams
        self.alpha_entropy = alpha_entropy
        self.gap = gap

        # Init network
        self.E = Encoder(fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1]//2**self.n_blocks, variational=True, gap=gap)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats//2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks,
                           gap=gap)

        if torch.cuda.is_available():
            self.E.cuda()
            self.Dec.cuda()

        if self.load_weigths:
            self.E.load_state_dict(torch.load(self.dir_results + 'encoder_weights.pth'))
            self.Dec.load_state_dict(torch.load(self.dir_results + 'decoder_weights.pth'))

        # Set parameters
        self.params = list(self.E.parameters()) + list(self.Dec.parameters())

        # Set losses
        if self.loss_reconstruction == 'l2':
            self.Lr = torch.nn.MSELoss(reduction='sum')
        elif self.loss_reconstruction == 'bce':
            self.Lr = torch.nn.BCEWithLogitsLoss(reduction='sum')

        self.Lkl = kl_loss

        # Set optimizers
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

        # Init additional variables and objects
        self.epochs = 0.
        self.iterations = 0.
        self.init_time = 0.
        self.lr_iteration = 0.
        self.lr_epoch = 0.
        self.kl_iteration = 0.
        self.kl_epoch = 0.
        self.H_iteration = 0.
        self.H_epoch = 0.
        self.i_epoch = 0.
        self.train_generator = []
        self.dataset_test = []
        self.metrics = {}
        self.aucroc_lc = []
        self.auprc_lc = []
        self.auroc_det = []
        self.lr_lc = []
        self.lkl_lc = []
        self.lae_lc = []
        self.H_lc = []
        self.auroc_det_lc = []
        self.refCam = 0.

    def train(self, train_generator, epochs, test_dataset):
        self.epochs = epochs
        self.init_time = timer()
        self.train_generator = train_generator
        self.dataset_test = test_dataset
        self.iterations = len(self.train_generator)

        # Loop over epochs
        for self.i_epoch in range(self.epochs):
            # init epoch losses
            self.lr_epoch = 0
            self.kl_epoch = 0.
            self.H_epoch = 0.

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, x_a, y_a) in enumerate(self.train_generator):
                #p = q

                # brain mask
                if 'BRATS' in train_generator.dataset.dir_datasets or\
                   'PhysioNet' in train_generator.dataset.dir_datasets:
                    x_mask = 1 - np.mean((x_n == 0).astype(np.int), 0)
                    if 'BRATS' in train_generator.dataset.dir_datasets:
                        x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 6, 6))).astype(x_mask.dtype)
                elif 'MVTEC' in train_generator.dataset.dir_datasets:
                    x_mask = np.zeros((1, 224, 224))
                    x_mask[:, 14:-14, 14:-14] = 1

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                z, z_mu, z_logvar, allF = self.E(x_n)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)
                if self.loss_reconstruction == 'l2':
                    xhat = torch.sigmoid(xhat)

                # Calculate criterion
                self.lr_iteration = self.Lr(xhat, x_n) / (self.train_generator.batch_size)  # Reconstruction loss
                self.kl_iteration = self.Lkl(mu=z_mu, logvar=z_logvar) / (self.train_generator.batch_size)  # kl loss

                # Init overall losses
                L = self.lr_iteration + self.alpha_kl * self.kl_iteration

                # ---- Compute Attention Homogeneization loss via Entropy

                am = torch.mean(allF[self.level_cams], 1)

                # Restore original shape
                am = torch.nn.functional.interpolate(am.unsqueeze(1),
                                                     size=(self.input_shape[-1], self.input_shape[-1]),
                                                     mode='bilinear',
                                                     align_corners=True)
                am = am.view((am.shape[0], -1))

                # Prepare mask with brain
                if 'BRATS' in train_generator.dataset.dir_datasets or\
                   'MVTEC' in train_generator.dataset.dir_datasets or\
                   'PhysioNet' in train_generator.dataset.dir_datasets:

                    x_mask = np.ravel(x_mask)
                    x_mask = torch.tensor(np.array(np.argwhere(x_mask > 0.5))).cuda().squeeze()
                    am = torch.index_select(am, dim=1, index=x_mask)

                # Probabilities
                p = torch.nn.functional.softmax(am.view((am.shape[0], -1)), dim=-1)
                # Mean entropy
                self.H_iteration = torch.mean(-torch.sum(p * torch.log(p + 1e-12), dim=(-1)))

                if self.i_epoch > self.pre_training_epochs:

                    if self.alpha_entropy > 0:
                        # Entropy Maximization
                        L += - self.alpha_entropy * self.H_iteration

                # Update weights
                L.backward()  # Backward
                self.opt.step()  # Update weights
                self.opt.zero_grad()  # Clear gradients

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.kl_epoch += self.kl_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.H_epoch += self.H_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.lr_lc.append(self.lr_epoch)
        self.lkl_lc.append(self.kl_epoch)
        self.H_lc.append(self.H_epoch)

        # Each x epochs, test models and plot learning curves
        if (self.i_epoch + 1) % self.epochs_to_test == 0:
            # Save weights
            torch.save(self.E.state_dict(), self.dir_results + 'encoder_weights.pth')
            torch.save(self.Dec.state_dict(), self.dir_results + 'decoder_weights.pth')

            # Evaluate
            if self.i_epoch > (self.pre_training_epochs - 50):
                # Make predictions
                Y, Scores, M, Mhat, X, Xhat = inference_dataset(self, self.dataset_test)

                # Input to dataset
                self.dataset_test.Scores = Scores
                self.dataset_test.Mhat = Mhat
                self.dataset_test.Xhat = Xhat

                # Evaluate anomaly detection
                auroc_det, auprc_det, th_det = evaluate_anomaly_detection(self.dataset_test.Y, self.dataset_test.Scores,
                                                                          dir_out=self.dir_results,
                                                                          range=[np.min(Scores)-np.std(Scores), np.max(Scores)+np.std(Scores)],
                                                                          tit='kl')
                acc = accuracy_score(np.ravel(Y), np.ravel((Scores > th_det)).astype('int'))
                fs = f1_score(np.ravel(Y), np.ravel((Scores > th_det)).astype('int'))

                metrics_detection = {'auroc_det': auroc_det, 'auprc_det': auprc_det, 'th_det': th_det, 'acc_det': acc,
                                     'fs_det': fs}
                print(metrics_detection)

                # Evaluate anomaly localization
                metrics, th = evaluate_anomaly_localization(self.dataset_test, save_maps=True, dir_out=self.dir_results)
                self.metrics = metrics

                # Save metrics as dict
                with open(self.dir_results + 'metrics.json', 'w') as fp:
                    json.dump(metrics, fp)
                print(metrics)

                # Plot learning curve
                self.plot_learning_curves()

                # Save learning curves as dataframe
                self.aucroc_lc.append(metrics['AU_ROC'])
                self.auprc_lc.append(metrics['AU_PRC'])
                self.auroc_det_lc.append(auroc_det)
                history = pd.DataFrame(list(zip(self.lr_lc, self.lkl_lc, self.H_lc, self.aucroc_lc, self.auprc_lc, self.auroc_det_lc)),
                                       columns=['Lrec', 'Lkl', 'H', 'AUCROC', 'AUPRC', 'AUROC_det'])
                history.to_csv(self.dir_results + 'lc_on_direct.csv')

        else:
            self.aucroc_lc.append(0)
            self.auprc_lc.append(0)
            self.auroc_det_lc.append(0)

    def predict_score(self, x):
        self.E.eval()
        self.Dec.eval()

        # brain mask
        if 'BRATS' in self.train_generator.dataset.dir_datasets or 'PhysioNet' in self.train_generator.dataset.dir_datasets:
            x_mask = 1 - (x == 0).astype(np.int)
            if 'BRATS' in self.train_generator.dataset.dir_datasets:
                x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 6, 6))).astype(x_mask.dtype)
            else:
                x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 3, 3))).astype(x_mask.dtype)
        elif 'MVTEC' in self.train_generator.dataset.dir_datasets:
            x_mask = np.zeros((1, x.shape[-1], x.shape[-1]))
            x_mask[:, 14:-14, 14:-14] = 1

        # Get reconstruction error map
        z, z_mu, z_logvar, f = self.E(torch.tensor(x).cuda().float().unsqueeze(0))
        xhat = torch.sigmoid(self.Dec(z)[0]).squeeze().detach().cpu().numpy()

        am = torch.mean(f[self.level_cams], 1)
        # Restore original shape
        mhat = torch.nn.functional.interpolate(am.unsqueeze(0), size=(self.input_shape[-1], self.input_shape[-1]),
                                               mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()

        # brain mask - Keep only brain region
        if 'BRATS' in self.train_generator.dataset.dir_datasets or \
           'PhysioNet' in self.train_generator.dataset.dir_datasets or \
           'MVTEC' in self.train_generator.dataset.dir_datasets:
            mhat[x_mask[0, :, :] == 0] = 0

        # Get outputs
        anomaly_map = mhat
        # brain mask - Keep only brain region
        if 'BRATS' in self.train_generator.dataset.dir_datasets or \
            'PhysioNet' in self.train_generator.dataset.dir_datasets or \
           'MVTEC' in self.train_generator.dataset.dir_datasets:
            score = np.std(anomaly_map[x_mask[0, :, :] == 1])
        else:
            score = np.std(anomaly_map)

        self.E.train()
        self.Dec.train()
        return score, anomaly_map, xhat

    def display_losses(self, on_epoch_end=False):

        # Init info display
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: ".format(self.i_epoch + 1, self.epochs,
                                                            self.i_iteration + 1, self.iterations)
        # Prepare values to show
        if on_epoch_end:
            lr = self.lr_epoch
            lkl = self.kl_epoch
            lH = self.H_epoch

            end = '\n'
        else:
            lr = self.lr_iteration
            lkl = self.kl_iteration
            lH = self.H_iteration

            end = '\r'

        # Init losses display
        info += "Reconstruction={:.4f} || KL={:.4f} || H={:.4f}".format(lr, lkl, lH)
        if self.train_generator.dataset.weak_supervision:
            info += " || H_a={:.4f}".format(lH_a)
        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ', ET=' + et, end=end)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            axes.plot(x, y, 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        plot_subplot(axes[0, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.lr_lc), "Reconstruc loss")
        plot_subplot(axes[0, 1], np.arange(self.i_epoch + 1) + 1, np.array(self.lkl_lc), "KL loss")
        plot_subplot(axes[1, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.H_lc), "H")
        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()

