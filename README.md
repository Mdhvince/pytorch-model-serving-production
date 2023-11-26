# Serving PyTorch model

#### My workflow for serving PyTorch models in production.

Directory structure:
```
.
├── experiments/
│   ├── dataset/ ...
│   ├── training/ ...
│   ├── inference/ ...
│   ├── model.py
│   ├── model_state_dict.pt
│   ├── ...
├── input_torch_model_archiver/
│   ├── model.py (copied from experiments/model.py)
│   ├── model_state_dict.pt (copied from experiments/model_state_dict.pt)
│   ├── model_handler.py
│   ├── ...
├── model_store/
```

- In the `experiment` directory, I am training the model and testing if the inference code works as expected.
- Once I am happy with the results, I copy the model weights `.pt`, the model architecture `model.py` and any other necessary files for inference to the `input_torch_model_archiver` directory.
- In the `input_torch_model_archiver` directory, I create a `model_handler.py` file which will be used by the `torch-model-archiver` to create a `.mar` file.
- I run the `torch-model-archiver` command to create a `.mar` file.
- I copy the `.mar` file to the `model_store` directory.
- Start the server `torchserve`.
- I send a request to the server to test if the server is running as expected.
- Stop the server `torchserve --stop`.

Copying, creating and running commands can be found in the `torchserve_script/*.sh` files.
