import warnings
warnings.warn('ignore')
import numpy as np
from rdkit.Chem import MolFromSmiles
from features import (atom_features,bond_features,
                        one_of_k_encoding, one_of_k_encoding_unk)
# import pysnooper as ps

degrees = [0, 1, 2, 3, 4, 5]
class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type
    #@ps.snoop()
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))# gain node

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees} # {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    #@ps.snoop()
    def __init__(self,ntype,features,rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    #@ps.snoop()
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)
    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph
def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep
# smiles = 'CC(C)O'
# graph = graph_from_smiles(smiles)
# q = ['a', 'b', 'c']  
# q.append(q)
# print(q[1])


# smiles = 'C(O)CC'
# array_rep = array_rep_from_smiles(smiles)
# print(array_rep['atom_features'].shape[0])
# print(array_rep)
# # data = np.zeros((122,array_rep['atom_features'].shape[1]))
# # for i,j in enumerate(array_rep['atom_features']):
# #     data[i] = j
# # #print(data.shape)
# # print(array_rep['atom_features'])
# def extract_bondfeatures_of_neighbors_by_degree(array_rep):
#     bond_features_by_atom_by_degree = []
#     for degree in degrees:
#         bond_features = array_rep['bond_features']
#         #print(bond_features.shape)
#         bond_neighbors_list = array_rep[('bond_neighbors', degree)]
#         summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
#         bond_features_by_atom_by_degree.append(summed_bond_neighbors)
#     return bond_features_by_atom_by_degree
# array_rep = array_rep_from_smiles('CC(=O)OCCC1=CC=CC=C1')
# print(array_rep)
# #print(np.array(array_rep['atom_features']).astype('float32'))
# summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
# atom_features = array_rep['atom_features'].astype('float32')
# atom_features = np.zeros((15,array_rep['atom_features'].shape[1]))
# for i,j in enumerate(array_rep['atom_features']):
#     atom_features[i] = j
# def connectivity_to_Matrix(array_rep, total_num_features,degree):#需修改
#     total_num = []
#     mat = np.zeros((total_num_features, total_num_features),'float32') 
#     if degree ==1:   
#         for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
#             mat[i,x] = 1        
#         return mat
#     else:
#         for i in range(degree):
#             atom_neighbors_list = array_rep[('atom_neighbors',i)].astype('int32')
#             total_num.append(len(atom_neighbors_list))
#         total_num = sum(total_num)
#         for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
#             mat[total_num + i,x] = 1
#         return mat
# def bond_features_by_degree(total_atoms,summed_degrees,degree):
#     mat = np.zeros((total_atoms,6),'float32')
#     total_num = []
#     if degree ==1:
#         for i,x in enumerate(summed_degrees[1]):
#             mat[i] = x
#         return mat
#     else:
#         for i in range(degree):
#             total_num.append(len(summed_degrees[i]))
#         total_num = sum(total_num)
#         for i,x in enumerate(summed_degrees[degree]):
#             mat[total_num + i] = x
#         return mat
# true_summed_degree = []
# for degree in range(len(degrees)):
#     atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
#     #print(atom_neighbors_list[0].dtype)
#     if len(atom_neighbors_list)==0:
#         atom_neighbor_matching_matrix = np.zeros((atom_features.shape[0], atom_features.shape[0]),'float32') 
#         true_summed_degree.append(np.zeros((atom_features.shape[0], 6),'float32'))
#     #print(atom_features)
#     #this matrix is used by every layer to match and sum all neighboring updated atom features to the atoms
#     else:
#         atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, atom_features.shape[0],degree)
#         true_summed_degree.append(bond_features_by_degree(atom_features.shape[0],summed_degrees,degree))
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     #print(atom_neighbor_matching_matrix)
#     #print(np.dot(atom_neighbor_matching_matrix,atom_features))
# print(summed_degrees)

