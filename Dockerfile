FROM pytorch/torchserve:latest-cpu

RUN python3 -m pip install --upgrade pip
RUN pip install nltk==3.8.1

RUN mkdir -p /home/model-server/nltk_data
RUN python3 -c "import nltk; nltk.download('wordnet', download_dir='/home/model-server/nltk_data')"

COPY input_torch_model_archiver/config.ini /home/model-server/
COPY input_torch_model_archiver/model.py /home/model-server/
COPY input_torch_model_archiver/model_state_dict.pt /home/model-server/
COPY input_torch_model_archiver/model_handler.py /home/model-server/
COPY input_torch_model_archiver/word2idx.pkl /home/model-server/

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

RUN mkdir -p /home/model-server/model_store

RUN torch-model-archiver \
    --model-name text_classifier_endpoint \
    --version 1.0 \
    --model-file /home/model-server/model.py \
    --serialized-file /home/model-server/model_state_dict.pt \
    --handler /home/model-server/model_handler.py \
    --extra-files /home/model-server/config.ini,/home/model-server/word2idx.pkl \
    --export-path /home/model-server/model_store

CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--model-store", "/home/model-server/model_store", \
     "--models", "text_classifier_endpoint=text_classifier_endpoint.mar"]
