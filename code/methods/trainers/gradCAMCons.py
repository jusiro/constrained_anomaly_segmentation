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


class AnomalyDetectorGradCamCons:
    def __init__(self, dir_results, item=['flair'], zdim=32, lr=1 * 1e-4, input_shape=(1, 224, 224), epochs_to_test=25,
                 load_weigths=False, n_blocks=5, dense=True, loss_reconstruction='bce', alpha_ae=0, alpha_kl=1,
                 pre_training_epochs=0, level_cams=-4, t=25, p_activation_cam=0.2,
                 expansion_loss_penalty='l2', avg_grads=True, gap=False):

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
        self.t = t
        self.p_activation_cam = p_activation_cam
        self.expansion_loss_penalty = expansion_loss_penalty
        self.alpha_ae = alpha_ae
        self.avg_grads = avg_grads
        self.gap = gap
        self.scheduler = False

        # Init network
        self.E = Encoder(fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1] // 2 ** self.n_blocks, variational=True, gap=gap)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats // 2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks,
                           spatial_dim=self.input_shape[1] // 2 ** self.n_blocks,
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
        self.i_epoch = 0.
        self.lae_iteration = 0.
        self.lae_epoch = 0.
        self.train_generator = []
        self.dataset_test = []
        self.metrics = {}
        self.aucroc_lc = []
        self.auprc_lc = []
        self.auroc_det = []
        self.lr_lc = []
        self.lkl_lc = []
        self.lae_lc = []
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
            self.lae_epoch = 0.

            if self.scheduler and self.expansion_loss_penalty == 'log_barrier' and self.i_epoch > self.pre_training_epochs:
                self.t = self.t * 1.01
                print(self.t, end='\n')

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, x_a, y_a) in enumerate(self.train_generator):
                # p = q

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

                # ---- Compute Attention expansion loss

                # Compute grad-cams
                gcam = grad_cam(allF[self.level_cams], torch.sum(z_mu), normalization='sigm',
                                avg_grads=True)
                # Restore original shape
                gcam = torch.nn.functional.interpolate(gcam.unsqueeze(1),
                                                       size=(self.input_shape[-1], self.input_shape[-1]),
                                                       mode='bilinear',
                                                       align_corners=True).squeeze()
                self.lae_iteration = torch.mean(gcam)

                if self.i_epoch > self.pre_training_epochs:
                    # Compute attention expansion loss
                    if self.expansion_loss_penalty == 'l1':  # L1
                        lae = torch.mean(torch.abs(-torch.mean(gcam, (-1)) + 1 - self.p_activation_cam))
                    elif self.expansion_loss_penalty == 'l2':  # L2
                        lae = torch.mean(torch.sqrt(torch.pow(-torch.mean(gcam, (-1)) + 1 - self.p_activation_cam, 2)))
                    elif self.expansion_loss_penalty == 'log_barrier':
                        z = -torch.mean(gcam, (1, 2)).unsqueeze(-1) + 1
                        lae = log_barrier(z - self.p_activation_cam, t=self.t) / self.train_generator.batch_size

                    # Update overall losses
                    L += self.alpha_ae * lae.squeeze()

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
                self.lae_epoch += self.lae_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.lr_lc.append(self.lr_epoch)
        self.lkl_lc.append(self.kl_epoch)
        self.lae_lc.append(self.lae_epoch)

        # Each x epochs, test models and plot learning curves
        if (self.i_epoch + 1) % self.epochs_to_test == 0:
            # Save weights
            torch.save(self.E.state_dict(), self.dir_results + 'encoder_weights.pth')
            torch.save(self.Dec.state_dict(), self.dir_results + 'decoder_weights.pth')

            # Evaluate
            if self.i_epoch > (0):
                # Make predictions
                Y, Scores, M, Mhat, X, Xhat = inference_dataset(self, self.dataset_test)

                # Input to dataset
                self.dataset_test.Scores = Scores
                self.dataset_test.Mhat = Mhat
                self.dataset_test.Xhat = Xhat

                # Evaluate anomaly detection
                auroc_det, auprc_det, th_det = evaluate_anomaly_detection(self.dataset_test.Y, self.dataset_test.Scores,
                                                                          dir_out=self.dir_results,
                                                                          range=[np.min(Scores) - np.std(Scores),
                                                                                 np.max(Scores) + np.std(Scores)],
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
                history = pd.DataFrame(
                    list(zip(self.lr_lc, self.lkl_lc, self.lae_lc, self.aucroc_lc, self.auprc_lc, self.auroc_det_lc)),
                    columns=['Lrec', 'Lkl', 'lae', 'AUCROC', 'AUPRC', 'AUROC_det'])
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

        # Compute gradients-cams
        gcam = grad_cam(f[self.level_cams], torch.sum(z_mu), normalization='min_max',
                        avg_grads=self.avg_grads)

        # Restore original shape
        mhat = torch.nn.functional.interpolate(gcam.unsqueeze(0), size=(self.input_shape[-1], self.input_shape[-1]),
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
            lae = self.lae_epoch
            end = '\n'
        else:
            lr = self.lr_iteration
            lkl = self.kl_iteration
            lae = self.lae_iteration
            end = '\r'

        # Init losses display
        info += "Reconstruction={:.4f} || KL={:.4f} || Lae={:.8f} ".format(lr, lkl, lae)

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
        plot_subplot(axes[1, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.lae_lc), "AE loss")
        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()


def grad_cam(activations, output, normalization='relu_min_max', avg_grads=False, norm_grads=False):
    def normalize(grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads * torch.pow(l2_norm, -1)

    # Obtain gradients
    gradients = torch.autograd.grad(output, activations, grad_outputs=None, retain_graph=True, create_graph=True,
                                    only_inputs=True, allow_unused=True)[0]

    # Normalize gradients
    if norm_grads:
        gradients = normalize(gradients)

    # pool the gradients across the channels
    if avg_grads:
        gradients = torch.mean(gradients, dim=[2, 3])
        # gradients = torch.nn.functional.softmax(gradients)
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

    # weight activation maps
    '''
    if 'relu' in normalization:
        GCAM = torch.sum(torch.relu(gradients * activations), 1)
    else:
        GCAM = gradients * activations
        if 'abs' in normalization:
            GCAM = torch.abs(GCAM)
        GCAM = torch.sum(GCAM, 1)
    '''
    GCAM = torch.mean(activations, 1)

    # Normalize CAM
    if 'sigm' in normalization:
        GCAM = torch.sigmoid(GCAM)
    if 'min' in normalization:
        norm_value = torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM - norm_value
    if 'max' in normalization:
        norm_value = torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM * norm_value.pow(-1)
    if 'tanh' in normalization:
        GCAM = torch.tanh(GCAM)
    if 'clamp' in normalization:
        GCAM = GCAM.clamp(max=1)

    return GCAM

