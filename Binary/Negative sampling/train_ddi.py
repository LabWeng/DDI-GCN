# %%
import warnings
warnings.filterwarnings('ignore')
import os 
import matplotlib as plt
import tensorflow as tf 
import tensorflow.keras.optimizers as optimizers
from data_prep import *
from utils import *
from ddi_model import ddi_gcn_with_attention
import pandas as pd 
import numpy as np 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# %%
tr_dataset = pd.read_csv(r'tr_dataset.csv')
val_dataset = pd.read_csv(r'val_dataset.csv') # Cross one 
tst_dataset = pd.read_csv(r'tst_dataset.csv')
max_atom = 65
batch_size = 256
fp_depth = 8

# %%
steps_per_epoch = int(len(tr_dataset.index)/batch_size)
validation_steps = int(len(val_dataset.index)/batch_size)
tst_steps = int(len(tst_dataset.index)/batch_size)

# %%
# def sparse_categorical_crossentropy(y_true, y_pred):
#     return K.sparse_categorical_crossentropy(y_true, y_pred+1e-5)
model = ddi_gcn_with_attention(fp_depth = fp_depth, conv_width =128,max_atoms = max_atom,
                                            L2_reg = False, num_input_atom_features = 51,
                                            num_bond_features = 10)

# %%
logdir = r'callbacks_shareattweight_{}'.format(fp_depth)
filepath = r'weights.h5'
callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience=10, min_delta=1e-3),
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True,
                            mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5)
]

# %%
model.summary()

# %%
from sklearn.metrics import roc_auc_score,average_precision_score
def auc_score(y_true, y_pred):
    return tf.py_function(roc_auc_score,(y_true, y_pred),tf.float32)
def aupr_score(y_true, y_pred):
    return tf.py_function(average_precision_score,(y_true, y_pred),tf.float32)

# %%
optimizer = optimizers.Adam(lr = 0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics = ['accuracy',auc,aupr_score])

# %%
history1_3 = model.fit_generator(train_data_generator(tr_dataset,batch_size,max_atom),callbacks = callbacks,
                    steps_per_epoch =steps_per_epoch,epochs =100,validation_data = val_data_generator(val_dataset,batch_size,max_atom),validation_steps = validation_steps)

# %%
pd.DataFrame(history1_3.history).to_csv(r'history_depth_{}.history.csv'.format(fp_depth),index=False)

# %%
# %%
#Evaluate 
pred_res = model.predict_generator(train_data_generator(tst_dataset,batch_size,max_atom),steps=tst_steps,verbose=1)
def evaluates(pred,dataset):
    from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score
    predicts = np.where(pred>=0.5,1,0)
    label = dataset['DDI'].tolist()[:pred.shape[0]]
    acc = accuracy_score(predicts,label)
    auc = roc_auc_score(label,pred)
    aupr = average_precision_score(label,pred)
    return 'ACC:',round(acc,3),'AUC:',round(auc,3),'AUPR:',round(aupr,3)

# %%
print(evaluates(pred_res,tst_dataset))


