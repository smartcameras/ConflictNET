# Conflict-Intensity-Estimation
Estimate the level of verbal conflict from raw speech signals

An end-to-end CNN-LSTM architecture with attention mechanism using Keras with Tensorflow backend. 

Dataset used - SSPNet Conflict Corpus (http://www.dcs.gla.ac.uk/vincia/?p=270)

Paper - 

Rajan, Vandana, Alessio Brutti, and Andrea Cavallaro. "ConflictNET: End-to-End Learning for Speech-based Conflict Intensity Estimation." IEEE Signal Processing Letters 26.11 (2019): 1668-1672.
(https://ieeexplore.ieee.org/document/8850055)

# Procedure

1. Download the dataset from (http://www.dcs.gla.ac.uk/vincia/?p=270)

2. Create train, val and test split according to the following paper

Schuller, Björn, et al. "The INTERSPEECH 2013 computational paralinguistics challenge: Social signals, conflict, emotion, autism." Proceedings INTERSPEECH 2013, 14th Annual Conference of the International Speech Communication Association, Lyon, France. 2013.

3. Change lines 10,11 and 12 in 'dataLoad.py' by providing the train, val and test paths in your computer.

4. Run 'conflict_net.py'

# Model Summary

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 240000, 1)         0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 239995, 64)        448       
_________________________________________________________________
batch_normalization_4 (Batch (None, 239995, 64)        256       
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 29999, 64)         0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 29996, 128)        32896     
_________________________________________________________________
batch_normalization_5 (Batch (None, 29996, 128)        512       
_________________________________________________________________
max_pooling1d_5 (MaxPooling1 (None, 4999, 128)         0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 4996, 256)         131328    
_________________________________________________________________
batch_normalization_6 (Batch (None, 4996, 256)         1024      
_________________________________________________________________
max_pooling1d_6 (MaxPooling1 (None, 832, 256)          0         
_________________________________________________________________
average_pooling1d_2 (Average (None, 208, 256)          0         
_________________________________________________________________
lstm_3 (LSTM)                (None, 208, 128)          197120    
_________________________________________________________________
seq_self_attention_2 (SeqSel (None, 208, 128)          8257      
_________________________________________________________________
lstm_4 (LSTM)                (None, 64)                49408     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 421,314
Trainable params: 420,418
Non-trainable params: 896
_________________________________________________________________



