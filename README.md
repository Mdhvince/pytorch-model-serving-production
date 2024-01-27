# Serving PyTorch model

### Build
From the root directory  
```bash
docker build -t inference_image -f ./docker/api/Dockerfile .`
```

### Run
```bash
docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 --name inference_container inference_image`
```

### Send request
```bash
curl -X POST http://localhost:8080/predictions/text_classifier_endpoint -H 'Content-Type: application/json' -d '{ "data": "This is a test!" }'
```