# Conflict-Intensity-Estimation
Estimate the level of verbal conflict from raw speech signals

An end-to-end CNN-LSTM architecture with attention mechanism using Keras with Tensorflow backend. 

Dataset used - SSPNet Conflict Corpus (http://www.dcs.gla.ac.uk/~vincia/dataconflict/)

Paper - 

Rajan, Vandana, Alessio Brutti, and Andrea Cavallaro. "ConflictNET: End-to-End Learning for Speech-based Conflict Intensity Estimation." IEEE Signal Processing Letters 26.11 (2019): 1668-1672.
(https://ieeexplore.ieee.org/document/8850055)

![True versus Predicted Conflict Values](https://github.com/smartcameras/ConflictNET/blob/master/conflictnet.png)

# Procedure

1. Download the dataset from (http://www.dcs.gla.ac.uk/~vincia/dataconflict/)

2. Create train, val and test split according to the following paper

Schuller, Björn, et al. "The INTERSPEECH 2013 computational paralinguistics challenge: Social signals, conflict, emotion, autism." Proceedings INTERSPEECH 2013, 14th Annual Conference of the International Speech Communication Association, Lyon, France. 2013.

3. Change lines 10,11 and 12 in 'dataLoad.py' by providing the train, val and test paths in your computer.

4. Run 'conflict_net.py'

# Demo using a Greek political debate (from CONFER dataset)

[![ConflictNet Demo](https://img.youtube.com/vi/6AH-ITHsQbw/0.jpg)](https://www.youtube.com/watch?v=6AH-ITHsQbw)

CONFER dataset: https://ibug.doc.ic.ac.uk/resources/confer/
