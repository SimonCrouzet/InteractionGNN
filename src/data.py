import os, functools, math, csv, random, sys
import json as json
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch.utils.data

import torch
from torch_geometric.data import Data, Dataset, Batch, DataLoader
from src.utils import load_with_compression, euclidian_distance
import logging

from src.dataset_utils import *

#basic logging config
logging.basicConfig(filename="data.log",level=logging.DEBUG)


"""
Dataset class to handle files from protein interfaces.
Data are not loaded yet, it needs to be with get(index).

Returns:
	Dataset: A dataset containing protein data files
"""
class ProteinDataset(Dataset):
	def __init__(self, data_dir, threshold=10.0, random_seed=123,
				 device=torch.device('cpu'), nb_features=None, nb_classes=2, exclude_last=False, use_scr=True, specific_scr_fea=None):
		super().__init__()

		# Set device
		self.device = device

		# Set data directory
		assert os.path.isdir(data_dir), '{} does not exist!'.format(data_dir)
		self.data_dir = data_dir

		# Set number of features (from the sample) and number of classes
		self.nb_features = nb_features
		self.nb_classes = nb_classes

		# Param to exclude last feature of samples - for example if the sample files already contain scores from another algorithm
		self.exclude_last = exclude_last

		# Set specific SCR regions to use - otherwise, use all
		if specific_scr_fea is not None:
			if type(specific_scr_fea) is not list:
				specific_scr_fea = [specific_scr_fea]
			for scr_fea in specific_scr_fea:
				if scr_fea not in ['R', 'C', 'S']:
					raise ValueError('{} is not a valid SCR feature - must be \'R\', \'C\' or \'S\''.format(scr_fea))
		self.specific_scr_fea = specific_scr_fea

		self.process_samples()

		random.seed(random_seed)
		self.dist_calc  = euclidian_distance
		self.threshold = threshold
		self.use_scr = use_scr

	def process_samples(self, samples_select=None):
		# Process samples by splitting them by class
		self.pairs = {c:[] for c in range(self.nb_classes)}
		for p in os.listdir(self.data_dir):
			if os.path.isdir(os.path.join(self.data_dir, p)):
				for c in range(self.nb_classes):
					if os.path.isdir(os.path.join(self.data_dir, p, str(c))):
						# Store a tuple (protein pair, nb of samples) for each class
						self.pairs[c].append((p,len(os.listdir(os.path.join(self.data_dir, p, '{}'.format(c))))))

		# If samples were selected, keep only those
		if samples_select is not None:
			self.pairs = {c:[] for c in range(self.nb_classes)}
			for s_idx in range(len(samples_select)):
				for p,n in samples_select[s_idx].items():
					# Store a tuple (protein pair, nb of samples) for each class
					self.pairs[s_idx].append((p,n))

		# Process final samples with name of files containing them
		self.samples = {c:{} for c in range(self.nb_classes)}
		self.nb_samples = {c:0 for c in range(self.nb_classes)}
		self.samples_list = []
		for c in self.pairs.keys():
			for p,n in self.pairs[c]:
				# Recreate name of files from protein pair and class
				samples_p = [f for f in os.listdir(os.path.join(self.data_dir, p, '{}'.format(c)))]

				# Set limit (selected samples, or all)
				limit = min(n, len(samples_p))

				for i in range(limit):
					# Store dict of samples, per class and per protein pair
					self.samples[c][p] = samples_p[i]
					# Store count of sample per class
					self.nb_samples[c] += 1
					# Store list of samples with the filename, the class and the protein pair
					self.samples_list.append((samples_p[i],c,p))


	def reduce(self, nb_negative, nb_positive=-1):
		# Reduce the dataset to a specific number of positive/negative samples
		# Only for binary classification!
		if self.nb_classes != 2:
			raise NotImplementedError('ProteinDataset: reduce() is not implemented for non-binary classification')
		if nb_negative == -1 and nb_positive == -1:
			return

		if nb_positive == -1 or nb_positive > self.nb_samples[1]:
			warnings.warn('Reducing: Not enough positive samples in the dataset')
			nb_positive = self.nb_samples[1]
		if nb_negative == -1 or nb_negative > self.nb_samples[0]:
			warnings.warn('Reducing: Not enough negative samples in the dataset')
			nb_negative = self.nb_samples[0] 

		count_samples = {0:0, 1:0}

		# Keep only the first nb_positive/nb_negative samples
		pairs_neg_select = {p:0 for p,n in self.pairs[0]}
		pairs_pos_select = {p:0 for p,n in self.pairs[1]}
		# Select negative samples by protein pair
		for p, nb_s in self.pairs[0]:
			if count_samples[0] < nb_negative:
				if count_samples[0] + nb_s <= nb_negative:
					pairs_neg_select[p] = nb_s
				else:
					pairs_neg_select[p] = nb_negative - count_samples[0]
				count_samples[0] += pairs_neg_select[p]
		# Select positive samples by protein pair
		for p, nb_s in self.pairs[1]:
			if count_samples[1] < nb_positive:
				if count_samples[1] + nb_s <= nb_positive:
					pairs_pos_select[p] = nb_s
				else:
					pairs_pos_select[p] = nb_positive - count_samples[1]
				count_samples[1] += pairs_pos_select[p]
		# Process samples again with the new selection
		self.process_samples(samples_select=(pairs_neg_select, pairs_pos_select))

	def __len__(self):
		return sum([v for v in self.nb_samples.values()])

	def one_hot_rcs(self, df, col_idx):
		# Encode the RCS features in one-hot encoding
		if self.use_scr:
			unique_fea = pd.array(['R', 'C', 'S'])

			for u in unique_fea:
				df[u] = df.apply(lambda row: int(row[col_idx] == u), axis=1)

		# Remove the original column
		df = df.drop(col_idx, axis=1)
		return df

	def __getitem__(self, idxs):
		if type(idxs) is int:
			# One sample requested
			return self.get(idxs)
		elif type(idxs) is list or tuple:
			# Multiple samples requested
			return [self.get(idx) for idx in idxs if self.get(idx) is not None]
		else:
			raise ValueError('Dataset.__getitem__() can\'t return items from indexes of type ', type(idxs))

	def get_idx(self, idx):
		# Get a sample from its index
		return self.get(idx)

	def get(self, idx):
		# Compute sample from the stored informations

		# Extract filename, class and protein pair from the index
		sample, label, pair = self.samples_list[idx]
		path = os.path.join(self.data_dir, pair, str(label), sample)

		# Load the sample from its file
		df = pd.read_csv(path, sep=',',header=None, index_col=None)

		# Drop NaN columns (if any)
		df.dropna(axis=1, how='all', inplace=True)

		# Analyze size of the dataset
		nb_rows, nb_cols = df.shape[0], df.shape[1]			
		nb_features = nb_cols - 3

		atoms_features = []
		edges_from = []
		edges_to = []

		if nb_features != self.nb_features:
			raise ValueError('Dataset.get(): Wrong number of features in the dataset (expected {}, got {})'.format(self.nb_features, nb_features))

		if self.exclude_last:
			# Exclude last feature - for example if the sample files already contain scores from another algorithm
			df = df.drop(nb_cols-1, axis=1)
			nb_cols -= 1

		if df[3].dtype == np.dtype('object'):
			# Dataset is containing RCS regions - encode them
			if self.specific_scr_fea:
				# If specific RCS regions are requested, keep only those
				df = df[df[3].isin(self.specific_scr_fea)].reset_index(drop=True)
				nb_rows = df.shape[0]
			df = self.one_hot_rcs(df=df, col_idx=3)
		else:
			raise ValueError('SCR features should be in index 3')

		if len(df) == 0:
			warnings.warn('Dataset.get(): Empty data for sample {}'.format(sample))
			return None


		for i in range(nb_rows):
			# For each row (residue, atom or cubes), extract the features
			atoms_features.append(df.loc[i][3:].values)

			# Extract neighbours and build edges
			for j in range(nb_rows):
				if i != j:
					distance = self.dist_calc(df.loc[i][:3].values, df.loc[j][:3].values)
					if distance <= self.threshold:
						edges_from.append(i)
						edges_to.append(j)
		
		
		protein_atom_fea = torch.Tensor(np.array(atoms_features))  # Atom features
		if self.nb_features is not None:
			if self.use_scr:
				expected_features = self.nb_features + 2 # 1 RCS feature was encoded into a vector of length 3
			else:
				expected_features = self.nb_features - 1 # 1 RCS feature was dropped
			# Check if the features shape is correct
			if (not self.exclude_last and protein_atom_fea.shape[1] != expected_features) or (self.exclude_last and protein_atom_fea.shape[1] != expected_features - 1):
				logging.info('Dataset: Sample {}_{} from pair {} did not possess the number of features expected'.format(sample, label, pair))
				return None
		
		protein_edges_idx = torch.LongTensor(np.array([edges_from, edges_to]))  # Edge connectivity - shape [2, num_edges]

		# Class Data from torch_geometric.data, with additional attributes 'protein pair' and 'sample filename'
		return Data(x=protein_atom_fea, edge_index=protein_edges_idx, y=label, idx=idx, pair=pair, sample=os.path.splitext(sample)[0])


def StratifiedSplit(dataset, first_partition=1.0, second_partition=0.0, shuffle=False, seed=random.seed(), in_memory=False, nb_classes=2):
	# Custom stratified split function for the dataset

	dataset_len = len(dataset)

	if first_partition + second_partition != 1.0:
		raise ValueError('StratifiedSplit error: Sum of split percentages must be equals to 1.0.')

	data_dict = {}
	# Create "class => list of data" Dict
	# => dataset.get(i) is calculating and loading the data, so it is computationally heavy!
	for i in tqdm(range(dataset_len), desc='Splitting data...', disable=in_memory):
		if not in_memory: # Data need to be loaded - computationally heavy!
			data = dataset.get(i)
		else: # Data were already loaded - light
			data = dataset[i]
		
		if data is not None:
			if data.pair not in data_dict:
				data_dict[data.pair] = {k:[] for k in range(nb_classes)}
			data_dict[data.pair][data.y].append(data)

	# Build splits
	first_split, second_split = [], []
	
	np.random.seed(seed)

	acc_keys = [k for k in data_dict.keys()]
	split_pcount = 0
	split1_limit = len(acc_keys) * first_partition
	if shuffle:
		random.shuffle(acc_keys) # Shuffle keys
	for p in acc_keys:
		# Create first split
		if split_pcount < round(split1_limit) or first_partition == 1.0:
			for k in range(nb_classes):
				for s in data_dict[p][k]:
					first_split.append(s)
			split_pcount += 1
		# Create second split
		else:
			for k in range(nb_classes):
				for s in data_dict[p][k]:
					second_split.append(s)

	del data_dict  # To free the memory used

	# Shuffle again to shuffle classes within the sets
	if shuffle:
		random.shuffle(first_split)
		random.shuffle(second_split)

	return first_split, second_split


def reduce_split(split, limits={0:100, 1:100}, nb_classes=2):#
	# Reduce a split to a specific number of samples - e.g. to have a balanced validation set
	pruned_split = []
	count_reduced = {k:0 for k in range(nb_classes)}
	for data in split:
		if count_reduced[data.y] < limits[data.y]:
			pruned_split.append(data)
			count_reduced[data.y] += 1
	return pruned_split



class kFoldStratifiedCrossValidation:
	def __init__(self, dataset, nb_folds=5, nb_classes=2, shuffle=False, seed=random.seed()):
		# Custom stratified cross-validation function for the dataset
		if len(dataset) < nb_folds*2:
			raise ValueError('kFold error: please use kFold with enough data in the dataset.')

		self.dataset = dataset
		if shuffle:
			random.shuffle(dataset)
		self.nb_folds = nb_folds
		self.nb_classes = nb_classes
		self.dataset_len = len(dataset)

		np.random.seed(seed)
		self.shuffle = shuffle

		# Decide if the validation set as to be balanced or not
		self.folds = [[] for f in range(nb_folds)]

		# Build folds
		self.build_folds()

	def build_folds(self):
		data_dict = {}
		# Create "class => list of data" Dict
		for i in tqdm(range(self.dataset_len), desc='Constructing kFold...'):
			data = self.dataset[i]

			if data.pair not in data_dict:
				data_dict[data.pair] = {k:[] for k in range(self.nb_classes)}

			data_dict[data.pair][data.y].append(data)
		
		acc_keys = [k for k in data_dict.keys()]
		nb_samples = len(acc_keys)
		if self.shuffle:
			random.shuffle(acc_keys) # Shuffle keys
		nb_samples_per_fold = nb_samples / self.nb_folds
		if nb_samples < self.nb_folds:
			raise ValueError('kFold error: please use kFold with enough data in the dataset.')
		for i in range(nb_samples):
			i_fold = int(i / nb_samples_per_fold)
			if i_fold > self.nb_folds -1:
				raise ValueError('Unexpected bug during StratifiedSplit initialization')
			# Push sample into the fold
			for t,s in data_dict[acc_keys[i]].items():
				self.folds[i_fold].extend(s)

		# Shuffle folds
		for f in self.folds:
			if self.shuffle:
				random.shuffle(f)
		if self.shuffle:
			random.shuffle(self.folds)

		del data_dict  # To free the memory used

	def __iter__(self):
		# Iterate over folds
		self.val_idx = -1
		return self

	def __next__(self):  # return (train_set, val_set)
		if self.val_idx >= self.nb_folds-1:
			raise StopIteration
		
		self.val_idx = (self.val_idx + 1) % self.nb_folds

		# Create train and validation sets regarding the fold index
		train_set, val_set = [], []
		for i_fold in range(self.nb_folds):
			if i_fold == self.val_idx:
				val_set.extend(self.folds[i_fold])
			else:
				train_set.extend(self.folds[i_fold])

		return train_set, val_set

