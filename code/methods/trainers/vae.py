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


class AnomalyDetectorVAE:
    def __init__(self, dir_results, item=['flair'], zdim=32, lr=1*1e-4, input_shape=(1, 224, 224), epochs_to_test=25,
                 load_weigths=False, n_blocks=5, dense=True, context=False, bayesian=False,
                 loss_reconstruction='bce', restoration=False, alpha_kl=0.1):

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
        self.context = context
        self.bayesian = bayesian
        self.loss_reconstruction = loss_reconstruction
        self.restoration = restoration
        self.alpha_kl = alpha_kl

        # Init network
        self.E = Encoder(fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1]//2**self.n_blocks, variational=True, gap=False)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats//2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks,
                           gap=False)

        if torch.cuda.is_available():
            self.E.cuda()
            self.Dec.cuda()

        if self.load_weigths:
            self.E.load_state_dict(torch.load(self.dir_results + '/encoder_weights.pth'))
            self.Dec.load_state_dict(torch.load(self.dir_results + '/decoder_weights.pth'))

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
        self.train_generator = []
        self.dataset_test = []
        self.metrics = {}
        self.aucroc_lc = []
        self.auprc_lc = []
        self.lr_lc = []
        self.lkl_lc = []

    def train(self, train_generator, epochs, dataset_test):
        self.epochs = epochs
        self.init_time = timer()
        self.train_generator = train_generator
        self.dataset_test = dataset_test
        self.iterations = len(self.train_generator)

        # Loop over epochs
        for self.i_epoch in range(self.epochs):
            # init epoch losses
            self.lr_epoch = 0
            self.kl_epoch = 0.

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):

                if self.context:  # if context option, data augmentation to apply context
                    (x_n_context, _) = augment_input_batch(x_n.copy())
                    x_n_context = torch.tensor(x_n_context).cuda().float()

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                if not self.context:
                    z, z_mu, z_logvar, _ = self.E(x_n)
                else:
                    z, z_mu, z_logvar, _ = self.E(x_n_context)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)
                if self.loss_reconstruction == 'l2':
                    xhat = torch.sigmoid(xhat)

                # Calculate criterion
                self.lr_iteration = self.Lr(xhat, x_n) / self.train_generator.batch_size  # Reconstruction loss
                self.kl_iteration = self.Lkl(mu=z_mu, logvar=z_logvar)  # kl loss (averaged per spatial feature)

                # Init overall losses
                L = self.lr_iteration + self.alpha_kl * self.kl_iteration

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

            # Epoch-end processes
            self.on_epoch_end()

    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.lr_lc.append(self.lr_epoch)
        self.lkl_lc.append(self.kl_epoch)

        # Each x epochs, test models and plot learning curves
        if (self.i_epoch + 1) % self.epochs_to_test == 0:
            # Save weights
            torch.save(self.E.state_dict(), self.dir_results + 'encoder_weights.pth')
            torch.save(self.Dec.state_dict(), self.dir_results + 'decoder_weights.pth')

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

            # Plot learning curve
            self.plot_learning_curves()

            # Save learning curves as dataframe
            self.aucroc_lc.append(metrics['AU_ROC'])
            self.auprc_lc.append(metrics['AU_PRC'])
            history = pd.DataFrame(list(zip(self.lr_lc, self.lr_lc, self.aucroc_lc, self.auprc_lc)),
                                   columns=['Lrec', 'Lkl', 'AUCROC', 'AUPRC'])
            history.to_csv(self.dir_results + 'lc_on_direct.csv')

        else:
            self.aucroc_lc.append(0)
            self.auprc_lc.append(0)

    def predict_score(self, x):
        self.E.eval()
        self.Dec.eval()

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
        if self.restoration:  # restoration reconstruction
            mhat, xhat = self.restoration_reconstruction(x)
        elif self.bayesian:  # bayesian reconstruction
            mhat, xhat = self.bayesian_reconstruction(x)
        else:
            # Network forward
            z, z_mu, z_logvar, f = self.E(torch.tensor(x).cuda().float().unsqueeze(0))
            xhat = np.squeeze(torch.sigmoid(self.Dec(z)[0]).cpu().detach().numpy())
            # Compute anomaly map
            mhat = np.squeeze(np.abs(x - xhat))
            # mhat = np.squeeze((x - xhat))

        # Keep only brain region
        mhat[x_mask[0, :, :] == 0] = 0

        # Get outputs
        anomaly_map = mhat
        score = np.mean(anomaly_map)

        self.E.train()
        self.Dec.train()
        return score, anomaly_map, xhat

    def bayesian_reconstruction(self, x):

        N = 100
        p_dropout = 0.20
        mhat = np.zeros((self.input_shape[1], self.input_shape[2]))

        # Network forward
        z, z_mu, z_logvar, f = self.E(torch.tensor(x).cuda().float().unsqueeze(0))
        xhat = self.Dec(torch.nn.Dropout(p_dropout)(z))[0].cpu().detach().numpy()

        for i in np.arange(N):
            if z_mu is None:  # apply dropout to z
                mhat += np.squeeze(np.abs(np.squeeze(torch.sigmoid(
                    self.Dec(torch.nn.Dropout(p_dropout)(z))[0]).cpu().detach().numpy()) - x)) / N
            else:  # sample z
                mhat += np.squeeze(np.abs(np.squeeze(torch.sigmoid(
                    self.Dec(self.E.reparameterize(z_mu, z_logvar))[0]).cpu().detach().numpy()) - x)) / N
        return mhat, xhat

    def restoration_reconstruction(self, x):
        N = 300
        step = 1 * 1e-3
        x_rest = torch.tensor(x).cuda().float().unsqueeze(0)

        for i in np.arange(N):
            # Forward
            x_rest.requires_grad = True
            z, z_mu, z_logvar, f = self.E(x_rest)
            xhat = self.Dec(z)[0]

            # Compute loss
            lr = kornia.losses.total_variation(torch.tensor(x).cuda().float().unsqueeze(0) - torch.sigmoid(xhat))
            L = lr / (self.input_shape[1] * self.input_shape[2])

            # Get gradients
            gradients = torch.autograd.grad(L, x_rest, grad_outputs=None, retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True, allow_unused=True)[0]

            x_rest = x_rest - gradients * step
            x_rest = x_rest.clone().detach()
        xhat = np.squeeze(x_rest.cpu().numpy())

        # Compute difference
        mhat = np.squeeze(np.abs(x - xhat))

        return mhat, xhat

    def display_losses(self, on_epoch_end=False):

        # Init info display
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: ".format(self.i_epoch + 1, self.epochs,
                                                            self.i_iteration + 1, self.iterations)
        # Prepare values to show
        if on_epoch_end:
            lr = self.lr_epoch
            lkl = self.kl_epoch
            end = '\n'
        else:
            lr = self.lr_iteration
            lkl = self.kl_iteration
            end = '\r'
        # Init losses display
        info += "Reconstruction={:.4f} || KL={:.4f}".format(lr, lkl)
        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ', ET=' + et, end=end)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            axes.plot(x, y, 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(1, 2, figsize=(20, 15))
        plot_subplot(axes[0], np.arange(self.i_epoch + 1) + 1, np.array(self.lr_lc), "Reconstruc loss")
        plot_subplot(axes[1], np.arange(self.i_epoch + 1) + 1, np.array(self.lkl_lc), "KL loss")
        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()