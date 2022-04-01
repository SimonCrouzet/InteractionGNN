from copy import Error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Sequential, global_mean_pool, BatchNorm
from torch.nn import Linear, Dropout
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
import collections

# NamedTuple for storing the output of the model
Prediction = collections.namedtuple('Prediction', ['loss', 'pred', 'gt', 'correct'])


class GCN(torch.nn.Module):
    def __init__(self,
            dim,
            activation_function=nn.ELU,
            final_activation=nn.Softmax(dim=1),
            loss_function=F.cross_entropy,
            **kwargs):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # self.dim_atoms = kwargs.get('dim_atoms')
        # self.dim_bonds = kwargs.get('dim_bonds')

        # size of input/output node feature dimension, self-loops and symmetric normalization

        self.in_dim = dim
        self.loss_function = loss_function
        self.final_activation = final_activation

        self.define_architecture(dim)

    def define_architecture(self, dim):
        if dim > 16:
            self.model = Sequential('x, edge_index, batch', [
                (BatchNorm(in_channels=dim), 'x -> x'),
                (GCNConv(dim, 100), 'x, edge_index -> x'),
                nn.ELU(inplace=True),
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=100), 'x -> x'),
                (Dropout(p=0.5), 'x -> x'),
                (Linear(100, 20), 'x -> x'),
                nn.ELU(inplace=True),
                (BatchNorm(in_channels=20), 'x -> x'),
                (Dropout(p=0.2), 'x -> x'),
                (Linear(20, 2), 'x -> x'),
                ]
            )
        else:
            self.model = Sequential('x, edge_index, batch', [
                (BatchNorm(in_channels=dim), 'x -> x'),
                (GCNConv(dim, 20), 'x, edge_index -> x'),
                nn.ELU(inplace=True),
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=20), 'x -> x'),
                (Dropout(p=0.2), 'x -> x'),
                (Linear(20, 2), 'x -> x'),
                ]
            )

    def reinit_gcn(self, dim,
            activation_function=nn.ELU,
            final_activation=nn.Softmax(dim=1),
            loss_function=F.cross_entropy,
            **kwargs):
        # Use it to redefine a network without restarting the pipeline
        return GCN.__new__(dim, activation_function, final_activation, loss_function, **kwargs)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return self.model(x, edge_index, batch)

    # data of one batch as input of NN -> calculate loss
    def trainGraph(self, batch, optimizer, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)

        self.train()
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        out = self(batch)
        target = batch.y

        # Calculate loss
        loss = F.cross_entropy(out, target, reduction='mean', weight=weights) # combines log_softmax and nll_loss

        # Backward pass
        loss.backward()  # derive gradients
        optimizer.step() # update weights

        # Get final prediction
        out = self.final_activation(out) # Use Softmax to get probability
        prediction = out.argmax(dim=1)  # Use the class with highest probability.

        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)

    @torch.no_grad()
    def predictGraph(self, batch, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)

        self.eval()
        # Forward pass
        out = self(batch)
        target = batch.y

        # Calculate loss
        loss = F.cross_entropy(out, target, reduction='mean', weight=weights)

        # Get final prediction
        out = self.final_activation(out) # Use Softmax to get probability
        prediction = out.argmax(dim=1)  # Use the class with highest probability.

        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)


class LinearNetwork(torch.nn.Module):
    def __init__(self,
            dim,
            architecture_type=None, # Use it for AutoGL
            activation_function=nn.ELU,
            final_activation=nn.Softmax(dim=1),
            loss_function=F.cross_entropy,
            **kwargs):
        super(LinearNetwork, self).__init__()
        torch.manual_seed(12345)
        # self.dim_atoms = kwargs.get('dim_atoms')
        # self.dim_bonds = kwargs.get('dim_bonds')

        # size of input/output node feature dimension, self-loops and symmetric normalization

        self.in_dim = dim
        self.loss_function = loss_function

        if dim > 21:
            self.model = Sequential('x, edge_index, batch', [
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=dim), 'x -> x'),
                (Linear(dim, 20), 'x -> x'),
                (nn.ELU(inplace=True), 'x -> x'),
                (BatchNorm(in_channels=20), 'x -> x'),
                (Linear(20, 1), 'x -> x'),
                (nn.Sigmoid(), 'x -> x'),
                ]
            )
        else:
            self.model = Sequential('x, edge_index, batch', [
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=dim), 'x -> x'),
                (Linear(dim, 1), 'x -> x'),
                (nn.Sigmoid(), 'x -> x'),
                ]
            )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return self.model(x, edge_index, batch)

    # data of one batch as input of NN -> calculate loss
    def trainGraph(self, batch, optimizer, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)
        
        self.train()
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        out = self(batch)
        target = batch.y
        if out.shape[1] != 1:
            raise ValueError("Expected shape of probabilities should be (N, 1)")
        out = out[:, 0]

        # Calculate loss
        loss = F.binary_cross_entropy(out, target.float(), reduction='mean', weight=weights)

        # Backward pass
        loss.backward()  # derive gradients
        optimizer.step() # update params based on gradients

        # Get final prediction
        prediction = [1 if p >= 0.5 else 0 for p in out]
        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)

    @torch.no_grad()
    def predictGraph(self, batch, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)

        self.eval()
        # Forward pass
        out = self(batch)
        target = batch.y
        if out.shape[1] != 1:
            raise ValueError("Expected shape of probabilities should be (N, 1)")
        out = out[:, 0]

        # Calculate loss
        loss = F.binary_cross_entropy(out, target.float(), reduction='mean', weight=weights)

        # Get final prediction
        prediction = [1 if p >= 0.5 else 0 for p in out]
        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)


class GAT(torch.nn.Module):
    def __init__(self,
            dim,
            activation_function=nn.ELU,
            final_activation=nn.Softmax(dim=1),
            loss_function=F.cross_entropy,
            **kwargs):
        super(GAT, self).__init__()
        torch.manual_seed(12345)

        
        self.in_dim = dim
        self.loss_function = loss_function
        self.final_activation = final_activation

        self.define_architecture(dim)

    def define_architecture(self, dim):
        if dim > 16:
            self.model = Sequential('x, edge_index, batch', [
                (BatchNorm(in_channels=dim), 'x -> x'),
                (GATConv(dim, 100), 'x, edge_index -> x'),
                nn.ELU(inplace=True),
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=100), 'x -> x'),
                (Dropout(p=0.5), 'x -> x'),
                (Linear(100, 20), 'x -> x'),
                nn.ELU(inplace=True),
                (BatchNorm(in_channels=20), 'x -> x'),
                (Dropout(p=0.2), 'x -> x'),
                (Linear(20, 2), 'x -> x'),
                ]
            )
        else:
            self.model = Sequential('x, edge_index, batch', [
                (BatchNorm(in_channels=dim), 'x -> x'),
                (GATConv(dim, 20), 'x, edge_index -> x'),
                nn.ELU(inplace=True),
                (global_mean_pool, 'x, batch -> x'),
                (BatchNorm(in_channels=20), 'x -> x'),
                (Dropout(p=0.2), 'x -> x'),
                (Linear(20, 2), 'x -> x'),
                ]
            )

    def reinit_gat(self, dim,
            activation_function=nn.ELU,
            final_activation=nn.Softmax(dim=1),
            loss_function=F.cross_entropy,
            **kwargs):
        # Use it to redefine a network without restarting the pipeline
        return GAT.__new__(dim, activation_function, final_activation, loss_function, **kwargs)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return self.model(x, edge_index, batch)

    def trainGraph(self, batch, optimizer, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)
        
        self.train()
        # Clear gradients
        optimizer.zero_grad()  # clear gradients

        # Forward pass
        out = self(batch)
        target = batch.y

        # Calculate loss
        loss = F.cross_entropy(out, target, reduction='mean', weight=weights)

        # Backward pass
        loss.backward()  # derive gradients
        optimizer.step() # update weights

        # Get final prediction
        out = self.final_activation(out) # Use Softmax to get probability
        prediction = out.argmax(dim=1)  # Use the class with highest probability.
        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)

    @torch.no_grad()
    def predictGraph(self, batch, device=torch.device('cuda'), weights=None):
        # Move data to GPU if available
        batch = batch.to(device)

        self.eval()
        # Forward pass
        out = self(batch)
        target = batch.y

        # Calculate loss
        loss = F.cross_entropy(out, target, reduction='mean', weight=weights)

        # Get final prediction
        out = self.final_activation(out) # Use Softmax to get probability
        prediction = out.argmax(dim=1)  # Use the class with highest probability.

        # Calculate accuracy
        correct = sum([1 if p == t else 0 for p,t in zip(prediction, target)])
        return Prediction(loss, out, batch.y, correct)

