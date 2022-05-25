import torch
import os

# Import different methods
from methods.trainers.ae import *
from methods.trainers.vae import *
from methods.trainers.anoVAEGAN import *
from methods.trainers.fanoGAN import *
from methods.trainers.AMCons import *
from methods.trainers.gradCAMCons import *
from methods.trainers.histEqualization import *

torch.autograd.set_detect_anomaly(True)


class AnomalyDetectorTrainer:
    def __init__(self, dir_out, method, item=['flair'], zdim=32, dense=True, variational=False, n_blocks=5, lr=1*1e-4,
                 input_shape=(1, 224, 224), load_weigths=False, epochs_to_test=10, context=False, bayesian=False,
                 restoration=False, loss_reconstruction='bce', iteration=0, level_cams=-4,
                 alpha_kl=10, alpha_entropy=0., expansion_loss_penalty='log_barrier', p_activation_cam=0.2, t=25,
                 alpha_ae=10):

        # Init input variables
        self.dir_out = dir_out
        self.method = method
        self.item = item
        self.zdim = zdim
        self.dense = dense
        self.variational = variational
        self.n_blocks = n_blocks
        self.input_shape = input_shape
        self.load_weights = load_weigths
        self.epochs_to_test = epochs_to_test
        self.context = context
        self.bayesian = bayesian
        self.restoration = restoration
        self.loss_reconstruction = loss_reconstruction
        self.lr = lr
        self.level_cams = level_cams
        self.expansion_loss_penalty = expansion_loss_penalty
        self.alpha_kl = alpha_kl
        self.alpha_entropy = alpha_entropy
        self.p_activation_cam = p_activation_cam
        self.t = t
        self.alpha_ae = alpha_ae

        # Prepare results folders
        self.dir_results = dir_out + item[0] + '/iteration_' + str(iteration) + str('/')
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if not os.path.isdir(dir_out + item[0] + '/'):
            os.mkdir(dir_out + item[0] + '/')
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Create trainer
        if self.method == 'ae':
            self.method = AnomalyDetectorAE(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                            input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                            load_weigths=self.load_weights, n_blocks=self.n_blocks, dense=self.dense,
                                            context=self.context, bayesian=self.bayesian,
                                            loss_reconstruction=self.loss_reconstruction, restoration=self.restoration)
        elif self.method == 'vae':
            self.method = AnomalyDetectorVAE(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                             input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                             load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                             dense=self.dense,
                                             context=self.context, bayesian=self.bayesian,
                                             loss_reconstruction=self.loss_reconstruction,
                                             restoration=self.restoration,
                                             alpha_kl=self.alpha_kl)
        elif self.method == 'anoVAEGAN':
            self.method = AnomalyDetectorAnoVAEGAN(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                                   input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                                   load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                                   dense=self.dense,
                                                   context=self.context, bayesian=self.bayesian,
                                                   loss_reconstruction=self.loss_reconstruction,
                                                   restoration=self.restoration,
                                                   alpha_kl=self.alpha_kl)
        elif self.method == 'fanoGAN':
            self.method = AnomalyDetectorFanoGAN(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                                 input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                                 load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                                 dense=self.dense,
                                                 context=self.context, bayesian=self.bayesian,
                                                 loss_reconstruction=self.loss_reconstruction,
                                                 restoration=self.restoration)
        elif self.method == 'gradCAMCons':
            self.method = AnomalyDetectorGradCamCons(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                                     input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                                     load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                                     dense=self.dense, loss_reconstruction=self.loss_reconstruction,
                                                     pre_training_epochs=50, level_cams=self.level_cams, t=self.t,
                                                     p_activation_cam=self.p_activation_cam,
                                                     expansion_loss_penalty='log_barrier', alpha_ae=self.alpha_ae,
                                                     alpha_kl=self.alpha_kl)
        elif self.method == 'camCons':
            self.method = AnomalyDetectorAMCons(self.dir_results, item=self.item, zdim=self.zdim, lr=self.lr,
                                                input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                                load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                                dense=self.dense, loss_reconstruction=self.loss_reconstruction,
                                                pre_training_epochs=0, level_cams=self.level_cams,
                                                alpha_entropy=self.alpha_entropy,
                                                alpha_kl=self.alpha_kl)
        elif self.method == 'histEqualization':
            self.method = AnomalyDetectorHistEqualization(self.dir_results, item=self.item)

        else:
            print('Uncorrect specified method... ', end='\n')

    def train(self, train_generator, epochs, dataset_test):
        self.method.train(train_generator, epochs, dataset_test)