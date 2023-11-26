#!/bin/bash

# This script is used to create the model archive (.mar) file that will go into the model store
# first create model_store directory if it doesn't exist
mkdir -p model_store

torch-model-archiver \
    --model-name text_classifier_endpoint \
    --version 1.0 \
    --serialized-file input_torch_model_archiver/model_state_dict.pt \
    --model-file input_torch_model_archiver/model.py \
    --handler input_torch_model_archiver/model_handler.py \
    --extra-files input_torch_model_archiver/word2idx.pkl,input_torch_model_archiver/config.ini \
    --export-path model_store