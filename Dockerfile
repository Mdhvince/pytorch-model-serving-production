FROM ubuntu:latest

WORKDIR /

RUN git clone https://github.com/pytorch/serve.git
WORKDIR /serve
RUN python ./ts_scripts/install_dependencies.py

WORKDIR /

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt


RUN mkdir input_torch_model_archiver
COPY input_torch_model_archiver/config.ini input_torch_model_archiver/config.ini
COPY input_torch_model_archiver/model.py input_torch_model_archiver/model.py
COPY input_torch_model_archiver/model_state_dict.pt input_torch_model_archiver/model_state_dict.pt
COPY input_torch_model_archiver/model_handler.py input_torch_model_archiver/model_handler.py
COPY input_torch_model_archiver/word2idx.pkl input_torch_model_archiver/word2idx.pkl

RUN mkdir model_store

RUN torch-model-archiver \
    --model-name text_classifier_endpoint \
    --version 1.0 \
    --serialized-file input_torch_model_archiver/model_state_dict.pt \
    --handler input_torch_model_archiver/model_handler.py \
    --extra-files input_torch_model_archiver/word2idx.pkl,input_torch_model_archiver/config.ini \
    --export-path model_store \

EXPOSE 8080

CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "text_classifier_endpoint=text_classifier_endpoint.mar"]
