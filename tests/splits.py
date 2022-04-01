import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch_geometric.data as geom_data
from src.data import ProteinDataset, StratifiedSplit, kFoldStratifiedCrossValidation, reduce_split
from torch_geometric.data import Data, Dataset, Batch, DataLoader
from src.model import GCN, GAT
from src.utils import as_numpy, as_numpy_tuple, plot_roc_pr_curves
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from configparser import ConfigParser
from src.config import str_to_bool, str_to_model, read_config

import unittest


class TestStratifiedSplit(unittest.TestCase):
    def testIndependance(self):
        data_dir = './data/intermediate_m5_e30_nonorm_split1/'
        contains_pairs = True
        skiptest = False
        use_kfold = True
        nb_folds = 5


        seed = 123  # Seed to fix the simulation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Build dataset
        dataset = ProteinDataset(data_dir, random_seed=seed, device=device, nb_features=22)
        dataset.reduce(nb_negative=200, nb_positive=200)

        dataset_train, dataset_test = StratifiedSplit(dataset, first_partition=0.85, second_partition=0.15, shuffle=True)
        # dataset_test = reduce_split(dataset_test, nb_positive=100, nb_negative=100)
        train_len = (len(dataset_train)*(nb_folds -1)) / nb_folds
        val_len = (len(dataset_train)) / nb_folds
        test_len = len(dataset_test)

        print()
        print('Training on {} proteins.'.format(train_len))
        print('Validating on {} proteins.'.format(val_len))
        print('Testing on {} proteins.'.format(test_len))

        kFoldStrCV = kFoldStratifiedCrossValidation(dataset_train, nb_folds=nb_folds, nb_classes=2, shuffle=True)
        iter_kFold = iter(kFoldStrCV)
        range_fold = range(nb_folds)

        # Set tests
        all_train_pairs = [d.pair for d in dataset_train]
        test_pairs = [d.pair for d in dataset_test]

        for i_p in range(len(test_pairs)):
            with self.subTest(msg='Pair {}'.format(test_pairs[i_p])):
                self.assertNotIn(test_pairs[i_p], all_train_pairs)

        
        for k in range_fold:
            train_set, val_set = iter_kFold.__next__()
            train_pairs = [d.pair for d in train_set]
            val_pairs = [d.pair for d in val_set]
            for i_p in range(len(val_pairs)):
                with self.subTest(msg='Fold {} - Pair {}'.format(k, val_pairs[i_p])):
                    self.assertNotIn(val_pairs[i_p], train_pairs)

if __name__ == '__main__':
    unittest.main()
