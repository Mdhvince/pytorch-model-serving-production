#!/bin/bash

# Start torchserve

torchserve \
    --start \
    --ncs \
    --model-store model_store \
    --models text_classifier_endpoint=text_classifier_endpoint.mar
    # --ts-config config.properties \ we use the default config.properties file


# to check the status of torchserve and the endpoint we use the management API: curl http://localhost:8081/models
# more details on the served model can be found with: curl http://localhost:8081/models/text_classifier_endpoint