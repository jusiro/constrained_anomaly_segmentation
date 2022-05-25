import datetime
import kornia
import json
import torch

import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from timeit import default_timer as timer
from datasets.utils import augment_input_batch
from models.models import Encoder, Decoder, GradConCAEEncoder, GradConCAEDecoder
from evaluation.utils import *
from sklearn.metrics import accuracy_score, f1_score
from methods.losses.losses import kl_loss


class AnomalyDetectorGradCons:
    def __init__(self, dir_results, item=['flair'], zdim=32, lr=1*1e-4, input_shape=(1, 224, 224), epochs_to_test=25,
                 load_weigths=False, n_blocks=5, dense=True, loss_reconstruction='bce', alpha_gradloss=1,
                 n_target_filters=1, alpha_kl=10, variational=True):

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
        self.alpha_gradloss = alpha_gradloss
        self.n_target_filters = n_target_filters
        self.alpha_kl = alpha_kl
        self.variational = variational

        # Init network
        self.E = Encoder(fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1]//2**self.n_blocks, variational=self.variational)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats//2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks)
        '''
        # Init network
        self.E = GradConCAEEncoder(fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                                   spatial_dim=self.input_shape[1]//2**self.n_blocks)
        self.Dec = GradConCAEDecoder(fin=self.zdim, n_channels=self.input_shape[0],
                                     dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks)
        '''

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
        self.i_iteration = 0.
        self.iterations = 0.
        self.init_time = 0.
        self.lr_iteration = 0.
        self.lr_epoch = 0.
        self.lgrad_iteration = 0.
        self.lgrad_epoch = 0.
        self.i_epoch = 0.
        self.kl_iteration = 0.
        self.kl_epoch = 0.
        self.train_generator = []
        self.dataset_test = []
        self.metrics = {}
        self.aucroc_lc = []
        self.auprc_lc = []
        self.lr_lc = []
        self.lgrad_lc = []

    def train(self, train_generator, epochs, dataset_test):
        self.epochs = epochs
        self.init_time = timer()
        self.train_generator = train_generator
        self.dataset_test = dataset_test
        self.iterations = len(self.train_generator)
        # Init gradients module
        self.ref_grad = self.initialise()
        self.k = 0

        # Loop over epochs
        for self.i_epoch in range(self.epochs):
            # init epoch losses
            self.lr_epoch = 0
            self.lgrad_epoch = 0.
            self.kl_epoch = 0.

            # Loop over training dataset
            for self.i_iteration, (x_n, y_n, _, _) in enumerate(self.train_generator):

                # Move tensors to gpu
                x_n = torch.tensor(x_n).cuda().float()

                # Obtain latent space from normal sample via encoder
                z, z_mu, z_logvar, _ = self.E(x_n)

                # Obtain reconstructed images through decoder
                xhat, _ = self.Dec(z)

                # Calculate criterion
                self.lr_iteration = self.Lr(xhat, x_n) / (self.train_generator.batch_size)  # Reconstruction loss
                self.kl_iteration = self.alpha_kl * self.Lkl(mu=z_mu, logvar=z_logvar) / (self.train_generator.batch_size)  # kl loss

                # Calculate gradient loss
                # self.kl_iteration.backward(create_graph=True, retain_graph=True)
                # self.lr_iteration.backward(create_graph=True, retain_graph=True)

                grad_loss = 0.
                i = 0
                for module in self.iterlist():
                    if isinstance(module, torch.nn.Conv2d):
                        wrt = module.weight
                        #target_grad = wrt.grad
                        target_grad = torch.autograd.grad(self.kl_iteration, wrt, create_graph=True, retain_graph=True)[0]
                        if self.k > 0:
                            grad_loss += -1 * torch.nn.functional.cosine_similarity(target_grad.view(-1, 1), self.ref_grad[i].view(-1, 1) / self.k, dim=0).squeeze()
                        self.ref_grad[i] += target_grad.detach()
                        i += 1
                if self.k == 0:
                    self.lgrad_iteration = torch.tensor(1.).cuda().float()
                else:
                    self.lgrad_iteration = grad_loss / i  # Average over layers

                    # Get overall losses - we already computed gradients from Lr, so it is not neccesary again
                    # L = self.lr_iteration + self.lgrad_iteration * self.alpha_gradloss
                    L = self.lr_iteration + self.lgrad_iteration * self.alpha_gradloss

                    L.backward()  # Backward

                    # Update the reference gradient
                    i = 0
                    for module in self.iterlist():
                        if isinstance(module, torch.nn.Conv2d):
                            self.ref_grad[i] += module.weight.grad
                            i += 1

                # Update weights
                self.opt.step()  # Update weights
                self.opt.zero_grad()  # Clear gradients

                # Update k counter
                self.k += 1

                """
                ON ITERATION/EPOCH END PROCESS
                """

                # Display losses per iteration
                self.display_losses(on_epoch_end=False)

                # Update epoch's losses
                self.lr_epoch += self.lr_iteration.cpu().detach().numpy() / len(self.train_generator)
                self.lgrad_epoch += self.lgrad_iteration.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.lr_lc.append(self.lr_epoch)
        self.lgrad_lc.append(self.lgrad_epoch)

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

            # Evaluate anomaly detection
            auroc_det, auprc_det, th_det = evaluate_anomaly_detection(self.dataset_test.Y, self.dataset_test.Scores,
                                                                      dir_out=self.dir_results)
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
            history = pd.DataFrame(list(zip(self.lr_lc, self.lgrad_lc, self.aucroc_lc, self.auprc_lc)),
                                   columns=['Lrec', 'LgradCons', 'AUCROC', 'AUPRC'])
            history.to_csv(self.dir_results + 'lc_on_direct.csv')

        else:
            self.aucroc_lc.append(0)
            self.auprc_lc.append(0)

    def predict_score(self, x):
        self.E.eval()
        self.Dec.eval()

        # Prepare brain eroded mask
        x_mask = 1 - (x == 0).astype(np.int)
        x_mask = ndimage.binary_erosion(x_mask, structure=np.ones((1, 6, 6))).astype(x_mask.dtype)

        # Get reconstruction error map
        x_n = torch.tensor(x).cuda().float().unsqueeze(0)
        z, z_mu, z_logvar, _ = self.E(x_n)
        xhat = self.Dec(z)[0]

        # Calculate criterion
        lr_sample = self.Lr(xhat, x_n) / (self.input_shape[1] * self.input_shape[2])  # Reconstruction loss
        kl_sample = self.alpha_kl * self.Lkl(mu=z_mu, logvar=z_logvar)  # kl loss

        # Calculate gradient loss for anomaly score
        #lr_sample.backward(create_graph=True, retain_graph=True)

        score = 0.
        i = 0
        for module in self.iterlist():
            if isinstance(module, torch.nn.Conv2d):
                wrt = module.weight
                #target_grad = wrt.grad
                target_grad = torch.autograd.grad(kl_sample, wrt, create_graph=True, retain_graph=True)[0]
                if self.k > 0:
                    score += 1 * torch.nn.functional.cosine_similarity(target_grad.view(-1, 1),
                                                                       self.ref_grad[i].view(-1, 1) / self.k,
                                                                       dim=0).squeeze()
                i += 1

        score = - score.cpu().detach().numpy() / i  # Average over layers

        # Compute anomaly map
        xhat = torch.sigmoid(xhat).cpu().detach().numpy()
        mhat = np.squeeze(np.abs(x - xhat))

        # Keep only brain region
        mhat[x_mask[0, :, :] == 0] = 0

        # Get outputs
        anomaly_map = mhat

        self.opt.zero_grad()  # Clear gradients
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
            lgradCons = self.lgrad_epoch
            end = '\n'
        else:
            lr = self.lr_iteration
            lgradCons = self.lgrad_iteration
            end = '\r'
        # Init losses display
        info += "Reconstruction={:.4f} || gradCons={:.4f}".format(lr, lgradCons)
        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            axes.plot(x, y, 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(1, 2, figsize=(20, 15))
        plot_subplot(axes[0], np.arange(self.i_epoch + 1) + 1, np.array(self.lr_lc), "Reconstruc loss")
        plot_subplot(axes[1], np.arange(self.i_epoch + 1) + 1, np.array(self.lgrad_lc), "gradCons loss")
        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()

    def initialise(self):

        ref_grad = []
        n = 0
        for i in np.arange(0, len(list(self.E.children()))):  # For each block
            for module in list(list(self.E.children())[i].modules()):  # For each layer in the block
                if isinstance(module, torch.nn.Conv2d):
                    # Get weight
                    wrt = module.weight
                    # Init 0s tensor
                    grad_init_module = torch.zeros(wrt.shape).cuda().float()
                    # Incorportate to list
                    ref_grad.append(grad_init_module)

                    n += 1
                    if n == self.n_target_filters:
                        break

        return ref_grad

    def iterlist(self):
        '''
        layers = []
        n = 0
        for i in np.arange(0, len(list(self.E.children()))):  # For each block
            for module in list(list(self.E.children())[i].modules()):  # For each layer in the block
                if isinstance(module, torch.nn.Conv2d):
                    # Get layer
                    layers.append(module)

                    n += 1
                    if n == self.n_target_filters:
                        break
            if n == self.n_target_filters:
                break

        return layers
        '''

        layers = []
        n = 0
        for i in np.arange(0, len(list(self.E.children()))):  # For each block

            for module in list(list(self.E.children())[i].modules()):  # For each layer in the block
                if isinstance(module, torch.nn.Conv2d):
                    # Get layer
                    layers.append(module)

                    n += 1
                    if n == self.n_target_filters:
                        break
            if n == self.n_target_filters:
                break

        return layers