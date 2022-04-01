# InteractionGNN

Source code to run InteractionGNN on protein-protein interfaces

### Usage
1. Set the desired parameters in *config.ini*
2. in a bash shell, run `python interaction_gnn.py`

### Requirements
##### Packages:
 - Manual: install Pytorch, Pytorch-Geometric, Cuda-Toolkit, Scikit-Learn and the packages numpy pandas matplotlib lz4 and tqdm (`conda install -c pytorch -c pyg -c conda-forge python=3.9 numpy pandas matplotlib tqdm pytorch pyg scikit-learn cuda-toolkit lz4`)
 - All-in-one: Run `conda create --name interaction_gnn --file interaction_gnn.yml`
\
InteractionGNN is using [Pytorch-Geometric]([github.com/pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)).

##### Data files:
Should be in the folder data, displayed like the following example for binary classification:
\
```
InteractionGNN
|   interaction_gnn.py
|
|___src
|   |   ...
|
|___data
    |___protein_pair_1
    |   |___0
    |   |   |   file1
    |   |   |   file2
    |   |
    |   |___1
    |       |   file3
    |       |   file4
    |   
    |___protein_pair_2
    |   |___0
    |   |   |   file5
    |   |   |   file6
    |   |
    |   |___1
    |       |   file7
    |       |   file8
    ..........
```

### Citing
If you use this code, please cite the associated paper:
\
```Y. Mohseni Behbahani, S. Crouzet, E. Laine, A. Carbone, *Deep Local Analysis evaluates protein docking conformations with locally oriented cubes*```