import numpy as np 
import pandas as pd

'''
This .py generates datasets with low efficiency, just for model test,
TODO An efficient input Pipline with Tensorflow tf.Data API
'''
degrees = [0,1,2,3,4,5]

def data_random_splits(file_path,p_tr,p_tst,random_state = 0):
    """ This function is to split data file into training,validation and test datasets,and return their idx in the data file for futher split """
    n_drugs = len(pd.read_csv(file_path).index)
    if not (p_tr > 0 and p_tst > 0 and np.allclose(p_tr + p_tst, 1.0)):
        raise ValueError('Train/Val/Tst proportions must be positive and add up to 1.0.')
    n_tr, n_tst = int(p_tr *n_drugs), int(p_tst *n_drugs)
    perm = np.random.permutation(n_drugs)
    tr_idx, tst_idx = np.sort(perm[:n_tr]),np.sort(perm[n_tr:])
    return tr_idx,tst_idx

def prepare_datasets(file_path, tr_idx, val_idx = None, tst_idx = None):
    data = pd.read_csv(file_path).sample(frac=1,random_state=0)
    tr_data = data.loc[tr_idx].reset_index(drop=True)
    tr_data = (np.array(tr_data['Smiles'].tolist()),np.array(tr_data['targets'].tolist()))
    val_data = data.loc[val_idx].reset_index(drop=True)
    val_data = (np.array(val_data['Smiles'].tolist()),np.array(val_data['targets'].tolist()))
    tst_data = data.loc[tst_idx].reset_index(drop=True)
    tst_data = (np.array(tst_data['Smiles'].tolist()),np.array(tst_data['targets'].tolist()))
    return tr_data, val_data, tst_data
def keras_dataset_prep(file_path, tr_idx,tst_idx = None):
    data = pd.read_csv(file_path).sample(frac=1).reset_index(drop=True)
    tr_data = data.loc[tr_idx].reset_index(drop=True)
    tst_data = data.loc[tst_idx].reset_index(drop=True)
    return tr_data,tst_data
def train_on_cross_validation(tr_data_data_frame,tst_data_data_frame,val_size = 0.1):
    n_drugs = len(tr_data_data_frame.index)
    n_tr, n_val = int((1-val_size) *n_drugs), int(val_size *n_drugs)
    perm = np.random.permutation(n_drugs)
    tr_idx, val_idx = np.sort(perm[:n_tr]),np.sort(perm[n_tr:])
    tr_data = tr_data_data_frame.loc[tr_idx].reset_index(drop=True)
    val_data = tr_data_data_frame.loc[val_idx].reset_index(drop=True)
    return tr_data,val_data,tst_data_data_frame
def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)
    
    Returns:
    ----------
    
    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]
    
    e.g.:
    
    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]
    
    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]  
    
    """
    bond_features_by_atom_by_degree = []
    for degree in degrees:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree
def preprocess_data_set_for_Model(traindata, valdata, testdata, training_batchsize = 128, testset_batchsize = 128):
    
    train = _preprocess_data(traindata[0], traindata[1], training_batchsize)
    validation = _preprocess_data(valdata[0],  valdata[1],  testset_batchsize )
    test = _preprocess_data(testdata[0], testdata[1], testset_batchsize )

    return train, validation, test