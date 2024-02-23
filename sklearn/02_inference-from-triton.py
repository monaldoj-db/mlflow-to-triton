# Databricks notebook source
# MAGIC %md
# MAGIC #### First, ensure that you are actively running a Triton Inference Server hosting your model.
# MAGIC in this example we are assuming you followed the example in the README are running on `localhost`

# COMMAND ----------

# MAGIC %pip install tritonclient[all]

# COMMAND ----------

# MAGIC %md
# MAGIC #####Inference from the grpc endpoint

# COMMAND ----------

import tritonclient.grpc as triton_grpc

HOST = 'localhost'
PORT = 8001
TIMEOUT = 60
MODEL_NAME = "iris_onnx"
MODEL_VERSION = "1"
INPUT_NAME = "input_measurements"
OUTPUT_NAME = "label"

client = triton_grpc.InferenceServerClient(url=f'{HOST}:{PORT}')

# COMMAND ----------

import json

# parse the input example for data to test inference
input_example_json = """
{
  "columns": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "data": [
    [
      6.1,
      2.8,
      4.7,
      1.2
    ]
  ]
}
"""

input_example = json.loads(input_example_json)

input_columns = input_example['columns']
input_data = input_example['data'][0]
print(input_data)

# COMMAND ----------

def triton_predict(model_name, arr):
    triton_input = triton_grpc.InferInput(INPUT_NAME, arr.shape, 'FP32')
    triton_input.set_data_from_numpy(arr)
    triton_output = triton_grpc.InferRequestedOutput(OUTPUT_NAME)
    response = client.infer(model_name, model_version=MODEL_VERSION, inputs=[triton_input], outputs=[triton_output])
    return response.as_numpy(OUTPUT_NAME)

# COMMAND ----------

import numpy as np

triton_result = triton_predict(MODEL_NAME, np.array([input_data], dtype=np.float32))
print(triton_result)
