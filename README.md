# Ultimate-PUPPI
CERN Summer 2022. Project on pileup removal and secondary vertex prediction using Graph Neural Networks.

## Some helpful tips to figure around

Running multiclass_train_model.py trains the model and saves it to the model_dir.

Then, you can use the model to predict the probability of each class for a given event.

load_pred_save.py loads a trained model at a particular epoch, saves the predictions and plots various metrics

loss_functions.py has all the loss functions used

helper_functions.py is a compilation of all the helper functions used in various codes. It has most of the relevant imports as well, so importing it works in most cases

modify the home_dir.txt file to point to the directory of the place where you cloned the repository

data_rearrange.py is a script that rearranges the data so that vertices are in decreasing order of total pt.
data_process.py is a script that processes the data to be in the format required by the model.

They have to be run one after the other.