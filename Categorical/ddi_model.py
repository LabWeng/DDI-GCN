import pandas as pd 
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf 
from tensorflow.keras import layers,Input
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from utils import auc

degrees = [0,1,2,3,4,5]

def input_layer(degrees,input_name,max_atoms,num_input_atom_features = 51,num_bond_features=10):
    input_list = []
    input_list.append(Input(name='input_atom_features_'+ str(input_name), shape=(max_atoms,num_input_atom_features)))
    for degree in degrees:
        input_list.append(Input(name='atom_features_selector_matrix_degree_'+str(input_name)+ str(degree), shape=(max_atoms,max_atoms)))
        input_list.append(Input(name='bond_features_degree_'+ str(input_name)+str(degree), 
                                                            shape=(max_atoms,num_bond_features)))
    input_list.append(Input(name = 'atoms_by_order'+ str(input_name),shape=(max_atoms,max_atoms)))
    return input_list

class Co_Attention_Layer(layers.Layer):
    def __init__(self,mask_q,mask_a,mean_mask_q,mean_mask_a,graph_feat_size,k,**kwargs):
        self.mask_q = mask_q
        self.mask_a = mask_a
        self.mean_mask_q = mean_mask_q
        self.mean_mask_a= mean_mask_a
        self.k = k
        self.graph_feat_size = graph_feat_size
        super(Co_Attention_Layer, self).__init__(**kwargs)
    def build(self, input_shape):

        self.W_m = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_m',
                                      trainable=True)
        self.W_v = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_v',
                                      trainable=True)
        self.W_q = self.add_weight(shape=(self.k, self.graph_feat_size),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_q',
                                      trainable=True)
        self.W_h = self.add_weight(shape=(1,self.k),
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      name='W_h',
                                      trainable=True)
        super(Co_Attention_Layer, self).build(input_shape)
    def call(self, inputs):
        V_n,Q_n = inputs[0],inputs[1]
        V_0 = tf.tanh(tf.div(tf.reduce_sum(V_n,axis=-1,keep_dims=True),self.mean_mask_q))
        Q_0 = tf.tanh(tf.div(tf.reduce_sum(Q_n,axis=-1,keep_dims=True),self.mean_mask_a))
        M_0 = tf.multiply(V_0,Q_0)
        H_v = tf.multiply(tf.tanh(tf.matmul(self.W_v,V_n)),tf.tanh(tf.matmul(self.W_m,M_0)))
        H_q = tf.multiply(tf.tanh(tf.matmul(self.W_q,Q_n)),tf.tanh(tf.matmul(self.W_m,M_0)))
        alpha_v = tf.nn.softmax(tf.matmul(self.W_h,H_v),axis=-1)
        alpha_q = tf.nn.softmax(tf.matmul(self.W_h,H_q),axis=-1)
        vector_v = tf.matmul(alpha_v,tf.transpose(V_n,[0,2,1]))
        vector_q = tf.matmul(alpha_q,tf.transpose(Q_n,[0,2,1]))
        
        return tf.squeeze(vector_v,1), tf.squeeze(vector_q,1),alpha_v,alpha_q

def ddi_gcn_with_attention(fp_depth = 4, conv_width = 20,max_atoms = 10, 
                                             L2_reg = 4e-4, num_input_atom_features = 62, 
                                             num_bond_features = 6, batch_normalization = True,K_for_weight=65): 
    """
    fp_length   # Usually neural fps need far fewer dimensions than morgan.
    fp_depth     # The depth of the network equals the fingerprint radius.
    conv_width   # Only the neural fps need this parameter.
    """
    #inputs = input_layer(degrees,0,max_atoms=max_atoms)
    # atom_features = Input(name='input_atom_features', shape=(max_atoms,num_input_atom_features))
    # print(atom_features)
    atom_features = Input(name='input_atom_features',shape=(max_atoms,num_input_atom_features))
    matrix_degree = []
    selector_matrix_degree_0 = Input(name='atom_features_selector_matrix_degree_0',shape=(max_atoms,max_atoms))
    selector_matrix_degree_1 = Input(name='atom_features_selector_matrix_degree_1',shape=(max_atoms,max_atoms))
    selector_matrix_degree_2 = Input(name='atom_features_selector_matrix_degree_2',shape=(max_atoms,max_atoms))
    selector_matrix_degree_3 = Input(name='atom_features_selector_matrix_degree_3',shape=(max_atoms,max_atoms))
    selector_matrix_degree_4 = Input(name='atom_features_selector_matrix_degree_4',shape=(max_atoms,max_atoms))
    selector_matrix_degree_5 = Input(name='atom_features_selector_matrix_degree_5',shape=(max_atoms,max_atoms))
    atoms_order_matrix = Input(name = 'atoms_by_order',shape=(max_atoms,max_atoms))
    matrix_degree.append([selector_matrix_degree_0,selector_matrix_degree_1,selector_matrix_degree_2,selector_matrix_degree_3,selector_matrix_degree_4,selector_matrix_degree_5])
    bond_degree = []
    bond_features_degree_0 = Input(name='bond_features_degree_0',shape=(max_atoms,num_bond_features))
    bond_features_degree_1 = Input(name='bond_features_degree_1',shape=(max_atoms,num_bond_features))
    bond_features_degree_2 = Input(name='bond_features_degree_2',shape=(max_atoms,num_bond_features))
    bond_features_degree_3 = Input(name='bond_features_degree_3',shape=(max_atoms,num_bond_features))
    bond_features_degree_4 = Input(name='bond_features_degree_4',shape=(max_atoms,num_bond_features))
    bond_features_degree_5 = Input(name='bond_features_degree_5',shape=(max_atoms,num_bond_features))
    bond_degree.append([bond_features_degree_0,bond_features_degree_1,bond_features_degree_2,bond_features_degree_3,bond_features_degree_4,bond_features_degree_5])
    inputs =[atom_features,selector_matrix_degree_0,bond_features_degree_0,selector_matrix_degree_1,bond_features_degree_1,selector_matrix_degree_2,bond_features_degree_2,selector_matrix_degree_3,bond_features_degree_3,selector_matrix_degree_4,bond_features_degree_4,selector_matrix_degree_5,
            bond_features_degree_5,atoms_order_matrix]
    # all_outputs_to_fingerprint = []
    num_atom_features = num_input_atom_features
    all_atom_features = []
    for i in range(fp_depth):
        activations_by_degree = []
        for degree in degrees:
            atom_features_of_previous_layer_this_degree = tf.matmul(matrix_degree[0][degree],atom_features)
            merged_atom_bond_features = layers.Concatenate(axis=2)([atom_features_of_previous_layer_this_degree,bond_degree[0][degree]])
            merged_atom_bond_features._keras_shape = (None,max_atoms,num_atom_features+num_bond_features)
            activations = layers.Dense(conv_width, activation='relu',name='activations_{}_degree_{}'.format(i, degree),
                                        )(merged_atom_bond_features)
            activations_by_degree.append(activations)

        this_activations_tmp = layers.Dense(conv_width, activation='relu', name='layer_{}_activations'.format(i),
                                            )(atom_features) 
        if i==0:
            all_atom_features.append(this_activations_tmp)
        merged_neighbor_activations = tf.add_n(activations_by_degree)
        new_atom_features = layers.Add()([merged_neighbor_activations,this_activations_tmp])
        if batch_normalization:
            new_atom_features = tf.keras.layers.BatchNormalization(trainable=True)(new_atom_features)

        atom_features = new_atom_features
        num_atom_features = conv_width
        all_atom_features.append(atom_features) 
    gru_layer = tf.keras.layers.GRU(conv_width,return_sequences=True)
    bi_gru = tf.keras.layers.Bidirectional(gru_layer)
    scale_layer = tf.keras.layers.Dense(1,activation='tanh')
    stack_and_transpose_atom_features = tf.transpose(tf.stack(all_atom_features),[1,2,0,3])
    reshaped_atom_features = tf.reshape(stack_and_transpose_atom_features,[-1,len(all_atom_features),conv_width]) # [batch_size*num_atom,layers,conv_with]
    final_states = bi_gru(reshaped_atom_features) # [batch_size*num_atom,layers,conv_with*2]
    align_weights = scale_layer(final_states) # [batch_size*num_atom,layers,1] 
    scaled_align_weights = tf.nn.softmax(align_weights,axis=-2) # [batch_size*num_atom,layers,1] 
    output_features = tf.squeeze(tf.matmul(tf.transpose(scaled_align_weights,[0,2,1]),final_states),axis=-2) # [batch_size*num_atom,embedding]
    
    final_output_features = tf.reshape(output_features,[-1,max_atoms,conv_width*2])
    model = Model(inputs = inputs, outputs = final_output_features)
    inputs_a = input_layer(degrees,1,max_atoms=max_atoms,num_input_atom_features = num_input_atom_features,num_bond_features=num_bond_features)
    inputs_b = input_layer(degrees,2,max_atoms=max_atoms,num_input_atom_features = num_input_atom_features,num_bond_features=num_bond_features)
    attention_mask_for_inputs_q = Input(name = 'matrix_for_attention_mask_q',shape=(max_atoms,1))
    attention_mask_for_inputs_a = Input(name = 'matrix_for_attention_mask_a',shape=(max_atoms,1))
    mean_mask_q = Input(name = 'mean_for_mask_a',shape=(1,1))
    mean_mask_a = Input(name = 'mean_for_mask_b',shape=(1,1))
    durg_a,drug_b= inputs_a,inputs_b
    gcn_a,gcn_b = model(durg_a),model(drug_b)
    gcn_a = tf.keras.layers.Dense(conv_width)(gcn_a)
    gcn_b = tf.keras.layers.Dense(conv_width)(gcn_b)
    gcn_a,gcn_b = tf.transpose(gcn_a,[0,2,1]),tf.transpose(gcn_b,[0,2,1])
    gcn_a,gcn_b,*attention_weights= Co_Attention_Layer(attention_mask_for_inputs_q,
                                    attention_mask_for_inputs_a,mean_mask_q,mean_mask_a,conv_width,K_for_weight)([gcn_a,gcn_b])
    outputs = layers.Concatenate()([gcn_a,gcn_b])
    for hidden in [100,100,100]:
        outputs = layers.Dense(hidden,activation = 'relu',kernel_regularizer= tf.keras.regularizers.l2(L2_reg))(outputs)
    outputs = layers.Dense(106,activation = 'softmax')(outputs)
    models = Model(inputs=[inputs_a,inputs_b,attention_mask_for_inputs_q,\
                           mean_mask_q,mean_mask_a,
                           attention_mask_for_inputs_a], outputs=[outputs])
    return models
