# Databricks notebook source
# MAGIC %pip install onnx skl2onnx onnxruntime

# COMMAND ----------

import numpy as np
import mlflow.onnx
import onnx
import onnxruntime

# COMMAND ----------

# this run is from the classic sklearn iris classification example 
mlflow_run_id = "5b05c6573225478eb860b0184cf85433"
logged_model = "runs:/%s/model" % mlflow_run_id

# load the model from mlflow registry
sk_model = mlflow.sklearn.load_model(logged_model)

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

# test inference from native sklearn model
predictions = sk_model.predict([input_data])
print(predictions)

# COMMAND ----------

from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# convert sk_model to onnx format
# the "input_measurements" is an arbitrary name, but needs to be referenced later on triton server
# use {zipmap: False} to ensure a tensor is passed back, which is eventually necessary for triton server
iris_onnx = to_onnx(sk_model, initial_types=[('input_measurements', FloatTensorType([None, len(input_data)]))], options={"zipmap":False})

# COMMAND ----------

import onnxruntime as rt
import os
import glob

# save the onnx model
try:
  mlflow.onnx.save_model(iris_onnx, './iris_onnx/')
  print("model saved successfully")
except:
  files = glob.glob('./iris_onnx/*')
  for f in files:
    os.remove(f)
  os.rmdir('./iris_onnx/')
  mlflow.onnx.save_model(iris_onnx, './iris_onnx/')
  print("deleted and replaced old model")

# COMMAND ----------

# test onnx inference through onnxruntime python package
sess = rt.InferenceSession("./iris_onnx/model.onnx", providers=["CPUExecutionProvider"])

# input and output names are necessary for configuration with 
input_name = sess.get_inputs()[0].name
print("Input name:  ", input_name)
label_name = sess.get_outputs()[0].name
print("Output name: ", label_name)
pred = sess.run([label_name], {input_name: np.array([input_data], dtype=np.float32)})[0]
print("Prediction:  ", pred)

# COMMAND ----------

# MAGIC %md
# MAGIC #### You're now ready to export the ./iris_onnx/model.onnx file from databricks onto compute running Nvidia Triton Inference Server

# COMMAND ----------


