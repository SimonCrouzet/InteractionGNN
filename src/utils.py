import numpy as np
import torch
import json, os
import lz4.frame
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from sklearn.metrics import RocCurveDisplay
import shelve

from src.model import Prediction


def as_numpy(values):
    # Convert a Tensor or a list of Tensors to numpy structures
    if type(values) is torch.Tensor:
        return values.cpu().detach().numpy()
    elif type(values) is list or type(values) is tuple:
        output = [0 for i in range(len(values))]
        for i in range(len(values)):
            if type(values[i]) is torch.Tensor:
                output[i] = values[i].cpu().detach().numpy()
            else:
                output[i] = values[i]
        return output
    else:
        return values

def as_numpy_tuple(pred:Prediction):
    # Pass Prediction named tuple from the model to Prediction named tuple made of numpy arrays
    return Prediction(as_numpy(pred.loss), as_numpy(pred.pred), as_numpy(pred.gt), as_numpy(pred.correct))


def save_with_compression(obj, path, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC):
    # Save a python object to a compressed file
    with lz4.frame.open(path, mode='wb') as fp:
        obj_bytes = json.dumps(obj).encode('utf-8')
        compressed_obj = lz4.frame.compress(obj_bytes, compression_level=compression_level)
        fp.write(compressed_obj)


def load_with_compression(path):
    # Load a python object from a compressed file
    with lz4.frame.open(path, mode='r') as fp:
        output_compressed_data = fp.read()
        obj_bytes = lz4.frame.decompress(output_compressed_data)
        obj = json.loads(obj_bytes.decode('utf-8'))
        return obj




def euclidian_distance(coords_from, coords_to):
    # Compute the euclidian distance between two points
	if len(coords_from) != len(coords_to):
		raise ValueError('Euclidian distance: coords don\'t have the same shape')
	else:
		return np.sqrt(np.array([np.power(i-j, 2) for i,j in zip(coords_from, coords_to)]).sum())


def plot_roc_pr_curves(metrics, savepath, plot_roc_lines=False):
    # Plot meaningful ROC and PR curves
    if not plot_roc_lines:
        if len(np.unique(np.array([len(a) for a in metrics['tprs']]))) != 1:
            raise ValueError("Error while plotting ROC curves")

    # Plot ROC Curve...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot ROC baseline
    ax1.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    if plot_roc_lines: # Plot each ROC curve (one per fold)
        for fpr,tpr,auc in zip(metrics['fprs'], metrics['tprs'], metrics['aucs']):
            ax1.plot(
                fpr,
                tpr,
                label=r"Mean ROC (AUC = %0.2f)" % (auc),
                lw=1,
                alpha=0.8,
            )
    else:
        # Plot mean ROC curve (one curve for all folds)
        # Area between best and worst ROC curves are colored in grey
        mean_tpr = np.mean(metrics['tprs'], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = skmetrics.auc(metrics['base_fpr'], mean_tpr)
        std_auc = np.std(metrics['aucs'])
        ax1.plot(
            metrics['base_fpr'],
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        std_tpr = np.std(metrics['tprs'], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(
            metrics['base_fpr'],
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

    # Label, legend and set axe limits
    ax1.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver Operating Characteristic (ROC) Curve",
        xlabel='False Positive Rate',
        ylabel='True Positive Rate'
    )
    ax1.legend(loc="lower right")

    # Plot PR Curve...
    if len(metrics['precisions']) != len(metrics['recalls']):
        raise ValueError("Error while computing prediction / recall")
    else:
        nb_folds = len(metrics['precisions'])
        # Plot each PR curve (one per fold)
        for p,r,i in zip(metrics['precisions'], metrics['recalls'], range(nb_folds)):
            ax2.plot(
                r,
                p,
                label="fold {}".format(i),
                lw=1
            )
        # Label, legend and set axe limits
        ax2.legend(loc="lower right")
        ax2.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Precision Recall Curve",
            xlabel='Recall',
            ylabel='Precision'
        )

    # Save figure
    plt.savefig(savepath)
    plt.close()




def save_session(shelf_path):
    # Save the current session
    my_shelf = shelve.open(shelf_path,'n')

    for key in globals().keys():
        try:
            my_shelf[key] = globals()[key]
        except:
            pass

    my_shelf.close()

def load_session(shelf_path):
    # Load a session
    my_shelf = shelve.open(shelf_path)
    for key in my_shelf:
        try:
            globals()[key]=my_shelf[key]
        except:
            print('Not loaded:', key)
            pass

