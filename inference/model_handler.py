import os
import pickle
from configparser import ConfigParser
from typing import List, Dict

import torch
from ts.torch_handler.base_handler import BaseHandler

from model import RNNClassifier

"""
This script will be packaged with the model when using the model-archiver.
Torchserve will invoke the handle() function of this script for every request.
"""


class RNNClassifierHandler(BaseHandler):
    def __init__(self):
        super(RNNClassifierHandler, self).__init__()
        self.initialized = False
        self.seq_length = None
        self.word2idx = None

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # Read config file
        config_file = os.path.join(model_dir, "config.ini")
        if not os.path.isfile(config_file):
            raise RuntimeError("Missing the config.ini file")
        config = ConfigParser(inline_comment_prefixes="#")
        config.read(config_file)
        self.seq_length = config["DEFAULT"].getint("sequence_length")

        # Read word2idx file
        word2idx_path = os.path.join(model_dir, "word2idx.pkl")
        if not os.path.isfile(word2idx_path):
            raise RuntimeError("Missing the word2idx.pkl file")

        with open(word2idx_path, "rb") as pickle_file:
            self.word2idx = pickle.load(pickle_file)

        self.model = RNNClassifier(config, len(self.word2idx) + 1, self.device, model_pt_path, train=False)
        self.initialized = True

    def preprocess(self, request: List[Dict]):
        body = request[0].get("body")
        data = body.get("data")
        preprocessed_data = RNNClassifier.prepare(data, self.word2idx, self.seq_length, self.device)
        return preprocessed_data

    def inference(self, preprocessed_data, *args, **kwargs):
        response: Dict = RNNClassifier.predict(self.model, preprocessed_data)
        return [response]  # we can't return a dict, so we return a list with a dict inside

    def postprocess(self, inference_output):
        return inference_output

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)

        preprocessed_data = self.preprocess(data)
        model_output = self.inference(preprocessed_data)
        response: List = self.postprocess(model_output)
        return response
