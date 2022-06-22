import os
import json
import argparse
import numpy as np

from datasets.datasets import TestDataset, MultiModalityDataset, WSALDataGenerator, MVTECDataset
from methods.train import AnomalyDetectorTrainer
from evaluation.utils import *


def main(args):

    exp = {"dir_datasets": args.dir_datasets, "dir_out": args.dir_out, "load_weigths": args.load_weigths,
           "epochs": args.epochs, "item": args.item, "method": args.method, "input_shape": args.input_shape,
           "batch_size": args.batch_size, "lr": args.learning_rate, "zdim": args.zdim, "images_on_ram": True,
           "wkl": args.wkl, "wr": 1, "wadv": args.wadv, "wae": args.wae, "epochs_to_test": 50, "dense": args.dense,
           "channel_first": True, "normalization_cam": args.normalization_cam, "avg_grads": True, "t": args.t,
           "context": args.context, "level_cams": args.level_cams, "p_activation_cam": args.p_activation_cam,
           "bayesian": args.bayesian, "loss_reconstruction": "bce",
           "expansion_loss_penalty": args.expansion_loss_penalty, "restoration": args.restoration,
           "n_blocks": args.n_blocks, "alpha_entropy": args.wH}

    if not os.path.isdir('../data/results/'):
        os.mkdir('../data/results/')

    metrics = []
    for iteration in [0, 1, 2]:

        if 'BRATS' in exp['dir_datasets'] or 'PhysioNet' in exp['dir_datasets']:

            # Set test
            test_dataset = TestDataset(exp['dir_datasets'], item=exp['item'], partition='val',
                                       input_shape=exp['input_shape'],
                                       channel_first=True, norm='max', histogram_matching=True)

            # Set train data loader
            dataset = MultiModalityDataset(exp['dir_datasets'], exp['item'], input_shape=exp['input_shape'],
                                           channel_first=exp['channel_first'], norm='max', hist_match=True)

            train_generator = WSALDataGenerator(dataset, partition='train', batch_size=exp['batch_size'], shuffle=True)

        elif 'MVTEC' in exp['dir_datasets']:

            # Set test
            test_dataset = MVTECDataset(exp['dir_datasets'], exp['item'], input_shape=exp['input_shape'],
                                        channel_first=exp['channel_first'], norm='max',
                                        partition='test')

            # Set train data loader
            dataset = MVTECDataset(exp['dir_datasets'], exp['item'], input_shape=exp['input_shape'],
                                   channel_first=exp['channel_first'], norm='max',
                                   partition='train')

            train_generator = WSALDataGenerator(dataset, partition='train', batch_size=exp['batch_size'], shuffle=True)

        # Set trainer and train model
        trainer = AnomalyDetectorTrainer(exp['dir_out'], exp['method'], item=exp['item'], zdim=exp['zdim'],
                                         dense=exp['dense'], n_blocks=exp['n_blocks'],
                                         lr=exp['lr'], input_shape=exp['input_shape'], load_weigths=exp['load_weigths'],
                                         epochs_to_test=exp['epochs_to_test'], context=exp['context'],
                                         bayesian=exp['bayesian'], restoration=exp['restoration'],
                                         loss_reconstruction=exp['loss_reconstruction'],
                                         level_cams=exp['level_cams'],
                                         expansion_loss_penalty=exp['expansion_loss_penalty'],
                                         iteration=iteration,
                                         alpha_kl=exp['wkl'], alpha_entropy=exp["alpha_entropy"],
                                         p_activation_cam=exp["p_activation_cam"],
                                         alpha_ae=exp["wae"], t=exp["t"])

        # Save experiment setup
        with open(exp['dir_out'] + 'setup.json', 'w') as fp:
            json.dump(exp, fp)

        if not args.only_test:
            # Train
            trainer.train(train_generator, exp['epochs'], test_dataset)
        else:

            trainer.method.train_generator = train_generator
            trainer.method.dataset_test = test_dataset

            thresholod_with_percentile = False
            thresholod_with_valsubset = False
            if thresholod_with_percentile:
                # Predictions on normal dataset
                Y_t, Scores_t, M_t, Mhat_t, X_t, Xhat_t = inference_dataset(trainer.method, dataset)
                th = np.percentile(np.ravel(Mhat_t), 99)
            elif thresholod_with_valsubset:
                val_dataset = TestDataset(exp['dir_datasets'], item=exp['item'], partition='val',
                                          input_shape=exp['input_shape'],
                                          channel_first=True, norm='max', histogram_matching=True)
                # Make predictions
                _, Scores, _, Mhat, _, Xhat = inference_dataset(trainer.method, val_dataset)
                # Input to dataset
                val_dataset.Scores = Scores
                val_dataset.Mhat = Mhat
                val_dataset.Xhat = Xhat
                # Get threshold
                _, th = evaluate_anomaly_localization(val_dataset, save_maps=False,
                                                      dir_out=trainer.method.dir_results,
                                                      th=None)
            else:
                th = None

            # Make predictions
            Y, Scores, M, Mhat, X, Xhat = inference_dataset(trainer.method, test_dataset)

            # Input to dataset
            test_dataset.Scores = Scores
            test_dataset.Mhat = Mhat
            test_dataset.Xhat = Xhat

            metrics_i, th_i = evaluate_anomaly_localization(test_dataset, save_maps=False,
                                                            dir_out=trainer.method.dir_results,
                                                            th=th)
            print(metrics_i)
            trainer.method.metrics = metrics_i

        # Save overall metrics
        metrics.append(list(trainer.method.metrics.values()))

    # Compute average performance and save performance in dictionary
    metrics = np.array(metrics)
    metrics_mu = np.mean(metrics, 0)
    metrics_std = np.std(metrics, 0)

    labels = list(trainer.method.metrics.keys())
    metrics_mu = {labels[i]: metrics_mu[i] for i in range(0, len(labels))}
    metrics_std = {labels[i]: metrics_std[i] for i in range(0, len(labels))}

    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_avg_val.json', 'w') as fp:
        json.dump(metrics_mu, fp)
    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_std_val.json', 'w') as fp:
        json.dump(metrics_std, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Settings
    parser.add_argument("--dir_datasets", default="../data/BRATS_5slices/", type=str)
    parser.add_argument("--dir_out", default="../data/gradCAMCons/tests/", type=str)
    parser.add_argument("--method", default="gradCAMCons", type=str)
    parser.add_argument("--item", default=["flair"], type=list, nargs="+")
    parser.add_argument("--load_weigths", default=False, type=bool)
    parser.add_argument("--only_test", default=False, type=bool)
    # Hyper-params training
    parser.add_argument("--input_shape", default=[1, 224, 224], type=list)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    # Auto-encoder architecture
    parser.add_argument("--zdim", default=32, type=int)
    parser.add_argument("--dense", default=True, type=bool)
    parser.add_argument("--n_blocks", default=4, type=int)
    # Residual-based inference options
    parser.add_argument("--restoration", default=False, type=bool)
    parser.add_argument("--bayesian", default=False, type=bool)
    parser.add_argument("--context", default=False, type=bool)
    # Settings with variational AE
    parser.add_argument("--wkl", default=1, type=float)
    # Settings with discriminator
    parser.add_argument("--wadv", default=0., type=float)
    # GradCAMCons
    parser.add_argument("--wae", default=1e4, type=float)
    parser.add_argument("--p_activation_cam", default=1e-2, type=float)
    parser.add_argument("--expansion_loss_penalty", default="log_barrier", type=str)
    parser.add_argument("--t", default=10, type=int)
    parser.add_argument("--normalization_cam", default="sigm", type=str)
    # AMCons
    parser.add_argument("--wH", default=0., type=float)
    parser.add_argument("--level_cams", default=-4, type=float)

    args = parser.parse_args()
    main(args)
