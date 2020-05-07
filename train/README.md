## To Train
N.txt is used to track the number of trail. Please do not keep other .txt file in this folder.  

        python3 run.py

## Hyper Parameters
modify config.py, (optionally) set sanity = True to train on only 10 data
* fc_scale  : dividing number of parameters in fully-connected layers
* conv_scale: dividing number of parameters in convolutional layers
* weights   : weighting factor for unbalanced dataset
* lr	    : learning rate
* batch size


