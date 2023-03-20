import warnings
warnings.filterwarnings('ignore')
from data_prep import *
from mol_graph import array_rep_from_smiles
import numpy as np
import tensorflow as tf 
from tensorflow.keras import backend as K

degrees = [0,1,2,3,4,5]
def atoms_by_order(rdkit_ix,max_atom):
    order_matrix = np.zeros((max_atom,max_atom),'float32')
    for i,j in enumerate(rdkit_ix):
        order_matrix[j,i] = 1
    return order_matrix
def connectivity_to_Matrix(array_rep, total_num_features,degree):#
    total_num = []
    mat = np.zeros((total_num_features, total_num_features),'float32') 
    if degree == 0:   
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[i,x] = 1        
        return mat
    else:
        for i in range(degree):
            atom_neighbors_list = array_rep[('atom_neighbors',i)].astype('int32')
            total_num.append(len(atom_neighbors_list))
        total_num = sum(total_num)
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[total_num + i,x] = 1
        return mat
def bond_features_by_degree(total_atoms,summed_degrees,degree):
    mat = np.zeros((total_atoms,10),'float32')
    total_num = []
    if degree == 0:
        for i,x in enumerate(summed_degrees[0]):
            mat[i] = x
        return mat
    else:
        for i in range(degree):
            total_num.append(len(summed_degrees[i]))
        total_num = sum(total_num)
        for i,x in enumerate(summed_degrees[degree]):
            mat[total_num + i] = x
        return mat
def train_data_generator(input_dataframe,batchsize,max_atom):
    while True:
        #files = pd.read_csv(input_file_path)
        max_atom = max_atom
        smiles_a = input_dataframe['drug_A'].tolist()
        smiles_b = input_dataframe['drug_B'].tolist()
        labels = input_dataframe['DDI'].tolist()
        assert len(smiles_a) == len(smiles_b)
        N = len(smiles_a)
        for i in range(int(np.ceil(N*1./batchsize))):
            batch_dict = {'input_atom_features_1':[],
            'atom_features_selector_matrix_degree_10':[],
            'atom_features_selector_matrix_degree_11':[],
            'atom_features_selector_matrix_degree_12':[],
            'atom_features_selector_matrix_degree_13':[],
            'atom_features_selector_matrix_degree_14':[],
            'atom_features_selector_matrix_degree_15':[],
            'bond_features_degree_10':[],
            'bond_features_degree_11':[],
            'bond_features_degree_12':[],
            'bond_features_degree_13':[],
            'bond_features_degree_14':[],
            'bond_features_degree_15':[],
            'atoms_by_order1':[],
            'matrix_for_attention_mask_q':[],
            'input_atom_features_2':[],
            'atom_features_selector_matrix_degree_20':[],
            'atom_features_selector_matrix_degree_21':[],
            'atom_features_selector_matrix_degree_22':[],
            'atom_features_selector_matrix_degree_23':[],
            'atom_features_selector_matrix_degree_24':[],
            'atom_features_selector_matrix_degree_25':[],
            'bond_features_degree_20':[],
            'bond_features_degree_21':[],
            'bond_features_degree_22':[],
            'bond_features_degree_23':[],
            'bond_features_degree_24':[],
            'bond_features_degree_25':[],
            'atoms_by_order2':[],
            'matrix_for_attention_mask_a':[],
            'mean_for_mask_a':[],
            'mean_for_mask_b':[],
            }
            #atom_features = []
            labels_batch = labels[i*batchsize:min(N,(i+1)*batchsize)]
            for j in smiles_a[i*batchsize:min(N,(i+1)*batchsize)]:
                array_rep = array_rep_from_smiles(j)
                batch_dict['mean_for_mask_a'].append([[array_rep['atom_features'].shape[0]]])
                temp_atom_features = np.zeros((max_atom,array_rep['atom_features'].shape[1]))
                matrix_for_attention_mask = np.ones((max_atom,1))*float('-inf')
                for g,k in enumerate(array_rep['atom_features']):
                    temp_atom_features[g] = k
                    matrix_for_attention_mask[g] = 0
                summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
                batch_dict['input_atom_features_1'].append(temp_atom_features)
                for degree in degrees:
                    atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
                    #print(atom_neighbors_list[0].dtype)
                    if len(atom_neighbors_list)==0:
                        atom_neighbor_matching_matrix = np.zeros((temp_atom_features.shape[0], temp_atom_features.shape[0]),'float32') 
                        true_summed_degree = np.zeros((temp_atom_features.shape[0], 10),'float32')
                    else:
                        atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, temp_atom_features.shape[0],degree)
                        true_summed_degree = bond_features_by_degree(temp_atom_features.shape[0],summed_degrees,degree)
                    batch_dict['bond_features_degree_1'+str(degree)].append(true_summed_degree)
                    batch_dict['atom_features_selector_matrix_degree_1'+str(degree)].append(atom_neighbor_matching_matrix)
                order_matrix = atoms_by_order(array_rep['rdkit_ix'],max_atom)
                batch_dict['atoms_by_order1'].append(order_matrix)
                batch_dict['matrix_for_attention_mask_q'].append(matrix_for_attention_mask)
            for s in smiles_b[i*batchsize:min(N,(i+1)*batchsize)]:
                array_rep = array_rep_from_smiles(s)
                temp_atom_features = np.zeros((max_atom,array_rep['atom_features'].shape[1]))
                batch_dict['mean_for_mask_b'].append([[array_rep['atom_features'].shape[0]]])
                matrix_for_attention_mask = np.ones((max_atom,1))*float('-inf')
                for g,k in enumerate(array_rep['atom_features']):
                    temp_atom_features[g] = k
                    matrix_for_attention_mask[g] = 0
                summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
                batch_dict['input_atom_features_2'].append(temp_atom_features)
                for degree in degrees:
                    atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
                    #print(atom_neighbors_list[0].dtype)
                    if len(atom_neighbors_list)==0:
                        atom_neighbor_matching_matrix = np.zeros((temp_atom_features.shape[0], temp_atom_features.shape[0]),'float32') 
                        true_summed_degree = np.zeros((temp_atom_features.shape[0], 10),'float32')
                    else:
                        atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, temp_atom_features.shape[0],degree)
                        true_summed_degree = bond_features_by_degree(temp_atom_features.shape[0],summed_degrees,degree)
                    batch_dict['bond_features_degree_2'+str(degree)].append(true_summed_degree)
                    batch_dict['atom_features_selector_matrix_degree_2'+str(degree)].append(atom_neighbor_matching_matrix)
                order_matrix = atoms_by_order(array_rep['rdkit_ix'],max_atom)
                batch_dict['atoms_by_order2'].append(order_matrix)
                batch_dict['matrix_for_attention_mask_a'].append(matrix_for_attention_mask)
            batch_dict = {key:np.array(value,dtype='float32') for key,value in batch_dict.items()}
            yield (batch_dict,np.array(labels_batch,dtype= np.int32))
def val_data_generator(input_dataframe,batchsize,max_atom):
    while True:
        #files = pd.read_csv(input_file_path)
        max_atom = max_atom
        smiles_a = input_dataframe['drug_A'].tolist()
        smiles_b = input_dataframe['drug_B'].tolist()
        labels = input_dataframe['DDI'].tolist()
        assert len(smiles_a) == len(smiles_b)
        N = len(smiles_a)
        for i in range(int(np.ceil(N*1./batchsize))):
            batch_dict = {'input_atom_features_1':[],
            'atom_features_selector_matrix_degree_10':[],
            'atom_features_selector_matrix_degree_11':[],
            'atom_features_selector_matrix_degree_12':[],
            'atom_features_selector_matrix_degree_13':[],
            'atom_features_selector_matrix_degree_14':[],
            'atom_features_selector_matrix_degree_15':[],
            'bond_features_degree_10':[],
            'bond_features_degree_11':[],
            'bond_features_degree_12':[],
            'bond_features_degree_13':[],
            'bond_features_degree_14':[],
            'bond_features_degree_15':[],
            'atoms_by_order1':[],
            'matrix_for_attention_mask_q':[],
            'input_atom_features_2':[],
            'atom_features_selector_matrix_degree_20':[],
            'atom_features_selector_matrix_degree_21':[],
            'atom_features_selector_matrix_degree_22':[],
            'atom_features_selector_matrix_degree_23':[],
            'atom_features_selector_matrix_degree_24':[],
            'atom_features_selector_matrix_degree_25':[],
            'bond_features_degree_20':[],
            'bond_features_degree_21':[],
            'bond_features_degree_22':[],
            'bond_features_degree_23':[],
            'bond_features_degree_24':[],
            'bond_features_degree_25':[],
            'atoms_by_order2':[],
            'matrix_for_attention_mask_a':[],
            'mean_for_mask_a':[],
            'mean_for_mask_b':[],
            }
            #atom_features = []
            labels_batch = labels[i*batchsize:min(N,(i+1)*batchsize)]
            for j in smiles_a[i*batchsize:min(N,(i+1)*batchsize)]:
                array_rep = array_rep_from_smiles(j)
                temp_atom_features = np.zeros((max_atom,array_rep['atom_features'].shape[1]))
                batch_dict['mean_for_mask_a'].append([[array_rep['atom_features'].shape[0]]])
                matrix_for_attention_mask = np.ones((max_atom,1))*float('-inf')
                for g,k in enumerate(array_rep['atom_features']):
                    temp_atom_features[g] = k
                    matrix_for_attention_mask[g] = 0
                summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
                batch_dict['input_atom_features_1'].append(temp_atom_features)
                for degree in degrees:
                    atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
                    #print(atom_neighbors_list[0].dtype)
                    if len(atom_neighbors_list)==0:
                        atom_neighbor_matching_matrix = np.zeros((temp_atom_features.shape[0], temp_atom_features.shape[0]),'float32') 
                        true_summed_degree = np.zeros((temp_atom_features.shape[0], 10),'float32')
                    else:
                        atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, temp_atom_features.shape[0],degree)
                        true_summed_degree = bond_features_by_degree(temp_atom_features.shape[0],summed_degrees,degree)
                    batch_dict['bond_features_degree_1'+str(degree)].append(true_summed_degree)
                    batch_dict['atom_features_selector_matrix_degree_1'+str(degree)].append(atom_neighbor_matching_matrix)
                order_matrix = atoms_by_order(array_rep['rdkit_ix'],max_atom)
                batch_dict['matrix_for_attention_mask_q'].append(matrix_for_attention_mask)
                batch_dict['atoms_by_order1'].append(order_matrix)
            for s in smiles_b[i*batchsize:min(N,(i+1)*batchsize)]:
                array_rep = array_rep_from_smiles(s)
                temp_atom_features = np.zeros((max_atom,array_rep['atom_features'].shape[1]))
                batch_dict['mean_for_mask_b'].append([[array_rep['atom_features'].shape[0]]])
                matrix_for_attention_mask = np.ones((max_atom,1))*float('-inf')
                for g,k in enumerate(array_rep['atom_features']):
                    temp_atom_features[g] = k
                    matrix_for_attention_mask[g] = 0
                summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
                batch_dict['input_atom_features_2'].append(temp_atom_features)
                for degree in degrees:
                    atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
                    #print(atom_neighbors_list[0].dtype)
                    if len(atom_neighbors_list)==0:
                        atom_neighbor_matching_matrix = np.zeros((temp_atom_features.shape[0], temp_atom_features.shape[0]),'float32') 
                        true_summed_degree = np.zeros((temp_atom_features.shape[0], 10),'float32')
                    else:
                        atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, temp_atom_features.shape[0],degree)
                        true_summed_degree = bond_features_by_degree(temp_atom_features.shape[0],summed_degrees,degree)
                    batch_dict['bond_features_degree_2'+str(degree)].append(true_summed_degree)
                    batch_dict['atom_features_selector_matrix_degree_2'+str(degree)].append(atom_neighbor_matching_matrix)
                order_matrix = atoms_by_order(array_rep['rdkit_ix'],max_atom)
                batch_dict['matrix_for_attention_mask_a'].append(matrix_for_attention_mask)
                batch_dict['atoms_by_order2'].append(order_matrix)
            batch_dict = {key:np.array(value,dtype='float32') for key,value in batch_dict.items()}
            yield (batch_dict,np.array(labels_batch,dtype= np.int32))
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def for_visualization_input(smiles_a,smiles_b,max_atom):
    batch_dict = {'input_atom_features_1':[],
            'atom_features_selector_matrix_degree_10':[],
            'atom_features_selector_matrix_degree_11':[],
            'atom_features_selector_matrix_degree_12':[],
            'atom_features_selector_matrix_degree_13':[],
            'atom_features_selector_matrix_degree_14':[],
            'atom_features_selector_matrix_degree_15':[],
            'bond_features_degree_10':[],
            'bond_features_degree_11':[],
            'bond_features_degree_12':[],
            'bond_features_degree_13':[],
            'bond_features_degree_14':[],
            'bond_features_degree_15':[],
            'atoms_by_order1':[],
            'matrix_for_attention_mask_q':[],
            'input_atom_features_2':[],
            'atom_features_selector_matrix_degree_20':[],
            'atom_features_selector_matrix_degree_21':[],
            'atom_features_selector_matrix_degree_22':[],
            'atom_features_selector_matrix_degree_23':[],
            'atom_features_selector_matrix_degree_24':[],
            'atom_features_selector_matrix_degree_25':[],
            'bond_features_degree_20':[],
            'bond_features_degree_21':[],
            'bond_features_degree_21':[],
            'bond_features_degree_22':[],
            'bond_features_degree_23':[],
            'bond_features_degree_24':[],
            'bond_features_degree_25':[],
            'atoms_by_order2':[],
            'matrix_for_attention_mask_a':[],
            'mean_for_mask_a':[],
            'mean_for_mask_b':[],
            }
    array_rep_a = array_rep_from_smiles(smiles_a)
    temp_atom_features_a = np.zeros((max_atom,array_rep_a['atom_features'].shape[1]))
    batch_dict['mean_for_mask_a'].append([[array_rep_a['atom_features'].shape[0]]])
    matrix_for_attention_mask_a = np.ones((max_atom,1))*float('-inf')
    for g,k in enumerate(array_rep_a['atom_features']):
        temp_atom_features_a[g] = k
        matrix_for_attention_mask_a[g] = 0
    summed_degrees_a = extract_bondfeatures_of_neighbors_by_degree(array_rep_a)
    batch_dict['input_atom_features_1'].append(temp_atom_features_a)
    for degree in degrees:
        atom_neighbors_list = array_rep_a[('atom_neighbors', degree)].astype('int32')
        #print(atom_neighbors_list[0].dtype)
        if len(atom_neighbors_list)==0:
            atom_neighbor_matching_matrix = np.zeros((temp_atom_features_a.shape[0], temp_atom_features_a.shape[0]),'float32') 
            true_summed_degree = np.zeros((temp_atom_features_a.shape[0], 10),'float32')
        else:
            atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep_a, temp_atom_features_a.shape[0],degree)
            true_summed_degree = bond_features_by_degree(temp_atom_features_a.shape[0],summed_degrees_a,degree)
        batch_dict['bond_features_degree_1'+str(degree)].append(true_summed_degree)
        batch_dict['atom_features_selector_matrix_degree_1'+str(degree)].append(atom_neighbor_matching_matrix)
    order_matrix = atoms_by_order(array_rep_a['rdkit_ix'],max_atom)
    batch_dict['matrix_for_attention_mask_q'].append(matrix_for_attention_mask_a)
    batch_dict['atoms_by_order1'].append(order_matrix)
    array_rep_b = array_rep_from_smiles(smiles_b)
    temp_atom_features_b = np.zeros((max_atom,array_rep_b['atom_features'].shape[1]))
    batch_dict['mean_for_mask_b'].append([[array_rep_b['atom_features'].shape[0]]])
    matrix_for_attention_mask_b = np.ones((max_atom,1))*float('-inf')
    for g,k in enumerate(array_rep_b['atom_features']):
        temp_atom_features_b[g] = k
        matrix_for_attention_mask_b[g] = 0
    summed_degrees_b = extract_bondfeatures_of_neighbors_by_degree(array_rep_b)
    batch_dict['input_atom_features_2'].append(temp_atom_features_b)
    for degree in degrees:
        atom_neighbors_list = array_rep_b[('atom_neighbors', degree)].astype('int32')
        #print(atom_neighbors_list[0].dtype)
        if len(atom_neighbors_list)==0:
            atom_neighbor_matching_matrix = np.zeros((temp_atom_features_b.shape[0], temp_atom_features_b.shape[0]),'float32') 
            true_summed_degree = np.zeros((temp_atom_features_b.shape[0], 10),'float32')
        else:
            atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep_b, temp_atom_features_b.shape[0],degree)
            true_summed_degree = bond_features_by_degree(temp_atom_features_b.shape[0],summed_degrees_b,degree)
        batch_dict['bond_features_degree_2'+str(degree)].append(true_summed_degree)
        batch_dict['atom_features_selector_matrix_degree_2'+str(degree)].append(atom_neighbor_matching_matrix)
    order_matrix = atoms_by_order(array_rep_b['rdkit_ix'],max_atom)
    batch_dict['matrix_for_attention_mask_a'].append(matrix_for_attention_mask_b)
    batch_dict['atoms_by_order2'].append(order_matrix)
    batch_dict = {key:np.array(value,dtype='float32') for key,value in batch_dict.items()}
    return batch_dict

if __name__ == "__main__":

    c = for_visualization_input('CC(C)O','CCCCC',5)
    # print(c)
