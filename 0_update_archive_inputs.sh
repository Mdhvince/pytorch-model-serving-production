#!/bin/bash

# This script is used to update the archive_inputs with the latest
# - model (.pt) file
# - model architecture (.py) file
# - extra files (e.g. vocab files, etc.)

# These files will be used to later create the model archive (.mar) file that will go into the model store

echo "Updating archive_inputs..."

cp src/model.py archive_inputs/model.py
cp src/model_state_dict.pt archive_inputs/model_state_dict.pt
cp src/word2idx.pkl archive_inputs/word2idx.pkl
cp src/config.ini archive_inputs/config.ini

echo "archive_inputs updated!"

