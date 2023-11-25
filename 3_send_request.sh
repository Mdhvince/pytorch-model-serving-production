#!/bin/bash

# Send a request to the endpoint. Put an input sentence in a variable
SENTENCE="This movie had the best acting and the dialogue was so good. I loved it."

curl http://localhost:8080/predictions/text_classifier_endpoint \
    -H "Content-Type: application/json" \
    -d '{"data": "'"$SENTENCE"'"}'