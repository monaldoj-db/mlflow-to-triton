#### After you export your model to ONNX, follow these steps to pull and spin up a Triton Inference Server with the sklearn iris onnx model running for inference
** This cannot be done within Databricks, this can be done on any compute capable of running the triton container**

With docker installed, pull the Triton Inference Server Docker image:
```bash
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
```

Ensure that the `model.onnx` file is stored in the following directory structure:

```bash
<model-repository-path>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
  ```

  ... or for this example, simply:

```bash
model_repository/
├─ iris_onnx/
│  ├─ 1/
│  │  ├─ model.onnx
│  ├─ config.pbtxt
```

Note: the `config.pbtxt` is not necessary for this example. We let the saved ONNX configuration be inferred by Triton server. 

If the `model_repository` directory is saved properly, simply run the docker container, mapping in the model repository as a volume.

```bash
docker run -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
```

This starts the Triton Server, and should appear like this:
![triton-server-running](https://github.com/monaldoj-db/mlflow-to-triton/assets/158090744/0eda0153-ec3a-4b30-8fb3-a5e97e9e5492)


Now your model is being served to `http` (on port 8000) and `grpc` (on port 8001) endpoints.
