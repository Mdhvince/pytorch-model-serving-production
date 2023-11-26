#!/bin/bash

# This script is used to update the input_torch_model_archiver/ with the latest
# - model (.pt) file
# - model architecture (.py) file
# - extra files (e.g. vocab files, etc.)

# These files will be used to later create the model archive (.mar) file that will go into the model store

echo "Updating input_torch_model_archiver/ ..."

cp src/model.py input_torch_model_archiver/model.py
cp src/model_state_dict.pt input_torch_model_archiver/model_state_dict.pt
cp src/word2idx.pkl input_torch_model_archiver/word2idx.pkl
cp src/config.ini input_torch_model_archiver/config.ini

echo "input_torch_model_archiver/ updated!"

