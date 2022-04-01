import copy
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from src.data import ProteinDataset, StratifiedSplit, kFoldStratifiedCrossValidation, reduce_split
from torch_geometric.data import Data, Dataset, Batch, DataLoader
from src.model import GCN, GAT
from src.utils import as_numpy, as_numpy_tuple, plot_roc_pr_curves
from src.metrics import ScoreDataframe
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from src.config import str_to_bool, str_to_model, read_config
from sklearn.utils.class_weight import compute_class_weight

print('================= InteractionGNN =================')

#Read config.ini file
runinfo, modelinfo, runparams = read_config('config.ini')

# parameters

start_time = datetime.now()

# Configuration, from the config.ini file
NN_model_string = modelinfo["NN_model_string"]
NN_model = str_to_model(NN_model_string)  # GCN or GAT or Linear
save_models = str_to_bool(runinfo["save_models"])
epochs = int(runparams["epochs"])
batch_size = int(runparams["batch_size"])
nb_folds = int(runparams["nb_folds"])
model_no = runinfo["model_no"]  # id for results
data_dir = runinfo["data_dir"]  # data directory
nb_negative = int(modelinfo["nb_negative"])  # -1 means 'No limit for negative samples'
nb_positive = int(modelinfo["nb_positive"])  # -1 means 'No limit for positive samples'
use_kfold = str_to_bool(runparams["use_kfold"])
nb_features = int(modelinfo["nb_features"])
use_weights = str_to_bool(runparams["use_weights"])
# Specific SCR regions should be of format 'S,C,R' in the config file
specific_scr_fea = modelinfo['specific_scr_fea'].split(',')
balance_test_split = str_to_bool(modelinfo["balance_test_split"])
exclude_last = str_to_bool(modelinfo["exclude_last"])

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))
if not os.path.exists('results'):
    os.mkdir('results')
print('Initializing dataset from {}'.format(data_dir))

seed = 123  # Seed to fix the simulation
random.seed(seed)
lr = float(modelinfo["learning_rate"])  # Learning rate for optimizer

# normalize train -> remember mean, std to use for val and test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build dataset
dataset = ProteinDataset(data_dir, random_seed=seed, device=device, nb_features=nb_features, use_scr=True, specific_scr_fea=specific_scr_fea)

# Reduce dataset to desired nb of samples
dataset.reduce(nb_negative=100, nb_positive=100)

print('Dataset built with {} samples - {} features and {} classes'.format(dataset.__len__(), dataset.nb_features, dataset.nb_classes))

if dataset.nb_classes == 2:
    print('\tDataset contains {} P and {} N'.format(dataset.nb_samples[1], dataset.nb_samples[0]))

print('Loading Data...')

# TODO: Test parallelism to speed up data loading

# Creating sets
if use_kfold:
    print('\tCreating train/val/test sets with k-fold cross validation...')
    dataset_train, dataset_test = StratifiedSplit(dataset, first_partition=0.85, second_partition=0.15, shuffle=True)
    dataset_val = []
    train_len = (len(dataset_train)*(nb_folds -1)) / nb_folds
    val_len = (len(dataset_train)) / nb_folds
    test_len = len(dataset_test)
    # Create folds
    kFoldStrCV = kFoldStratifiedCrossValidation(dataset_train, nb_folds=nb_folds, nb_classes=2, shuffle=True)
    iter_kFold = iter(kFoldStrCV)
    range_fold = range(nb_folds)
else:
    print('\tCreating train/val/test sets (without CV)...')
    dataset_train, dataset_test_val = StratifiedSplit(dataset, first_partition=0.7, second_partition=0.3, shuffle=True)
    dataset_test, dataset_val = StratifiedSplit(dataset_test_val, first_partition=0.5, second_partition=0.5, shuffle=True, in_memory=True, nb_classes=dataset.nb_classes)
    del dataset_test_val  # Unused heavy data
    train_len = len(dataset_train)
    test_len = len(dataset_test)
    val_len = len(dataset_val)
    range_fold = range(1)

assert(len(dataset_train)+len(dataset_val) > len(dataset_test)), 'Test set is surprisingly large!'

# Balancing test split
if balance_test_split:
    print('\tBalancing test set...')
    dataset_test = reduce_split(dataset_test, limits={0:100, 1:100}, nb_classes=dataset.nb_classes)
    test_len = len(dataset_test)


dim = dataset_train[0].x.shape[1]

# Save pairs used in different sets
max_length = len(dataset_train)+len(dataset_val)
split_df = pd.DataFrame({'all_train':[d.pair for d in dataset_train+dataset_val],'test':[dataset_test[i].pair  if i < len(dataset_test) else '' for i in range(max_length)]})

print('Splits created with {} train/{} val/{} test samples'.format(train_len, val_len, test_len))

# create model
# Set neural_network architecture - to be changed on the flight
print('Instanciating Graph Neural Network architecture...')
kwargs = {}
base_model = NN_model(**kwargs, dim=dim)
models = [copy.deepcopy(base_model).to(device) for _ in range_fold]
print('\tModel {} instanciated on device {}: {}'.format(NN_model, device, models[0]))
#model.init_weights(torch.nn.init.xavier_normal_, 0.1)

# main loop
hist = {"train_loss": np.array([0.0 for e in range(epochs)]), "val_loss": np.array([0.0 for e in range(epochs)]), "val_acc": np.array([0.0 for e in range(epochs)])}
scores = ScoreDataframe()


metrics_val = {'fprs':[], 'tprs':[], 'precisions':[], 'recalls':[], 'aucs':np.zeros(nb_folds)}
for k in range_fold:
    print('=========')
    print(f'Fold {k + 1}:')
    print('=========')
    print()

    optimizer = torch.optim.Adam(models[k].parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.

    # Preparing test and val sets
    if use_kfold:
        train_set, val_set = iter_kFold.__next__()
    else:
        train_set, val_set = dataset_train, dataset_val

    # Compute weights for each class. Since validation loss is only informative, use weights from train set only.
    if use_weights:
        weights = torch.Tensor(compute_class_weight('balanced', classes=[0,1], y=[d.y for d in train_set])).to(device)
    else:
        weights = torch.Tensor(compute_class_weight(None, classes=[0,1], y=[d.y for d in train_set])).to(device)

    split_df.insert(len(split_df.columns), 'train_fold{}'.format(k), [train_set[i].pair if i < len(train_set) else '' for i in range(max_length)])
    split_df.insert(len(split_df.columns), 'val_fold{}'.format(k), [val_set[i].pair if i < len(val_set) else '' for i in range(max_length)])

    # Divide sets into batches
    graph_train_loader = geom_loader.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    graph_val_loader = geom_loader.DataLoader(val_set, batch_size=batch_size, shuffle=True)  # Additional loader if you want to change to a larger dataset

    # Running model
    for epoch in range(epochs):
        print('---------')
        print(f'epoch {epoch + 1}:')

        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        outputs, label = np.empty((0,dataset.nb_classes), dtype=np.float32), np.empty((0), dtype=np.int64) # for ROC PR plots of val dataset

        # Train
        for step, data in enumerate(graph_train_loader):
            # Moving data to GPU memory
            data = data.to(device)

            # Forward pass
            train_pred = as_numpy_tuple(models[k].trainGraph(data, optimizer, device, weights=weights))

            train_loss += train_pred.loss

            #Move back data to CPU and flush GPU memory
            if device.type != 'cpu':
                torch.cuda.empty_cache()

            # if save_models:
            #     scores.add_rows(conformation_names=data.pair, folds=[k for i in range(len(train_pred.gt))], scores=train_pred.pred[:, 1], time_of_predictions=[np.NaN for i in range(len(train_pred.gt))], targets=train_pred.gt, epoch=epoch)
        
        # Validation
        for step, data in enumerate(graph_val_loader):
            # Moving data to GPU memory
            data = data.to(device)

            # Forward pass
            val_pred = as_numpy_tuple(models[k].predictGraph(data, device))

            val_loss += val_pred.loss
            val_acc += val_pred.correct

            if save_models:
                scores.add_rows(conformation_names=data.pair, folds=[k for i in range(len(val_pred.gt))], scores=val_pred.pred[:, 1], time_of_predictions=[np.NaN for i in range(len(val_pred.gt))], targets=val_pred.gt, epoch=epoch)

            # data for plot  # TODO: Fix val plot for batch size > 1
            outputs = np.concatenate((outputs, val_pred.pred), axis=0)
            label = np.concatenate((label, val_pred.gt), axis=0)

            # Move back data to CPU and flush GPU memory
            if device.type != "cpu":
                torch.cuda.empty_cache()
        
        # Little bias here: loss are divided by number of batches, therefore the last batch (if smaller than batch_size) had more importance than the others.
        train_loss /= len(graph_train_loader)
        val_loss /= len(graph_val_loader)

        val_acc = val_acc / len(val_set)

        print('Training loss: ', train_loss)
        print('Val loss: ', val_loss)
        print('Val accuracy: %4.2f%%' % (100.0*float(val_acc)))
        print('---------')

        hist['train_loss'][epoch] += train_loss
        hist['val_loss'][epoch] += val_loss
        hist['val_acc'][epoch] += val_acc

        # save model
        # if save_models:
        #     torch.save(models[k], 'results/' + NN_model_string + '_' + model_no + '_model_epoch_' + str(epoch+1) + '.pt') # , PATH

    if dataset.nb_classes == 2:
        fpr, tpr, _ = roc_curve(y_true=label, y_score=outputs[:,1])
        auc = roc_auc_score(y_true=label, y_score=outputs[:,1])
        precision, recall, fscore = precision_recall_curve(y_true=label, probas_pred=outputs[:,1])
        metrics_val['fprs'].append(fpr)
        metrics_val['tprs'].append(tpr)
        metrics_val['aucs'][k] = auc
        metrics_val['precisions'].append(precision)
        metrics_val['recalls'].append(recall)
        plot_roc_pr_curves(metrics_val, 'results/roc_pr_val_curve_' + NN_model_string+ '_' + model_no + '.svg', plot_roc_lines=True)

if use_kfold:
    hist['train_loss'] = np.divide(hist['train_loss'], nb_folds)
    hist['val_loss'] = np.divide(hist['val_loss'], nb_folds)
    hist['val_acc'] = np.divide(hist['val_acc'], nb_folds)

# test NN using test dataset
print('--------------------------')
print('Testing on test dataset...')
# Divide set into batches
graph_test_loader = geom_loader.DataLoader(dataset_test, batch_size=1)

test_loss = 0
test_acc = 0
metrics_test = {'base_fpr':np.linspace(0, 1, len(dataset_test)), 'tprs':np.zeros((nb_folds, len(dataset_test))), 'precisions':[], 'recalls':[], 'aucs':np.zeros(nb_folds)}
for k in range_fold:
    outputs, label = np.empty((0,dataset.nb_classes), dtype=np.float32), np.empty((0), dtype=np.int64) # for ROC PR plots of val dataset

    for step, data in enumerate(graph_test_loader):
        # Moving data to GPU memory
        data = data.to(device)
        
        # Forward pass
        test_pred = as_numpy_tuple(models[k].predictGraph(data, device))

        test_loss += test_pred.loss
        test_acc += test_pred.correct
        outputs = np.concatenate((outputs, val_pred.pred), axis=0)
        label = np.concatenate((label, val_pred.gt), axis=0)

        if save_models:
            scores.add_rows(conformation_names=data.pair, folds=[-1 for i in range(len(test_pred.gt))], scores=test_pred.pred[:, 1], time_of_predictions=[np.NaN for i in range(len(test_pred.gt))], targets=test_pred.gt, epoch=-1)

        # Move back data to CPU and flush GPU memory
        if device.type != "cpu":
            torch.cuda.empty_cache()

    if dataset.nb_classes == 2:
        fpr, tpr, _ = roc_curve(y_true=label, y_score=outputs[:,1])
        auc = roc_auc_score(y_true=label, y_score=outputs[:,1])
        precision, recall, fscore = precision_recall_curve(y_true=label, probas_pred=outputs[:,1])
        interp_tpr = np.interp(metrics_test['base_fpr'], fpr, tpr)
        interp_tpr[0] = 0.0
        metrics_test['tprs'][k] = interp_tpr
        metrics_test['aucs'][k] = auc
        metrics_test['precisions'].append(precision)
        metrics_test['recalls'].append(recall)
        plot_roc_pr_curves(metrics_test, 'results/roc_pr_test_curve_' + NN_model_string+ '_' + model_no + '.svg')


test_loss /= (len(graph_test_loader) * nb_folds)
test_acc /= (len(dataset_test) * nb_folds)

print('Test loss: ', test_loss)
print('Test accuracy: %4.2f%%' % (100.0*float(test_acc)))
print('--------------------------')

# plot figure
fig, ax = plt.subplots(1,1)
ax.plot([e for e in range(0, epochs)], hist["train_loss"], label="train_loss")
ax.plot([e for e in range(0, epochs)], hist["val_loss"], label="val_loss")
ax.plot([e for e in range(0, epochs)], hist["val_acc"], label="val_acc")

#ax.set_xticks(np.arange(0, epochs, 1.0))
#ax.set_xticklabels(np.arange(1, epochs+1, 1))
ax.set_xlabel("epoch")
ax.legend()
fig.savefig('results/' + NN_model_string + '_' + model_no + '.svg')

# info of run
split_df.to_csv('results/splits_' + NN_model_string+ '_' + model_no + '.csv', mode='a', header=True)
scores.export('results/prediction_results_' + NN_model_string + '_' + model_no +'.csv')
f= open("results/" + NN_model_string + "_" + model_no + ".txt", "w+")
f.write("Data from " + data_dir + "\n")
f.write("NN_model: " + NN_model_string + "\n")
f.write("model: " + str(models[0]) + "\n")
f.write("epochs: " + str(epochs) + "\n")
f.write("batch size: " + str(batch_size) + "\n")
end_time = datetime.now()
f.write("duration: " + str(end_time-start_time) + "\n")
f.write("size train dataset: " + str(train_len) + "\n")
f.write("size validation dataset: " + str(val_len) + "\n")
f.write("size test dataset: " + str(test_len) + "\n")
# class proportions
f.close()

# time
print('Duration: {}'.format(end_time - start_time))