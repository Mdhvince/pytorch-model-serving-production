#!/bin/bash

# This script is used to create the model archive (.mar) file that will go into the model store
# first create model_store directory if it doesn't exist
mkdir -p model_store

torch-model-archiver \
    --model-name text_classifier_endpoint \
    --version 1.0 \
    --serialized-file archive_inputs/model_state_dict.pt \
    --model-file archive_inputs/model.py \
    --handler archive_inputs/model_handler.py \
    --extra-files archive_inputs/word2idx.pkl,archive_inputs/config.ini \
    --export-path model_store