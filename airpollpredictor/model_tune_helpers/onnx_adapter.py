"""Wrapper for ONNX to save/load models"""

# pylint: disable=E0401

import numpy
import onnx
from onnxmltools.convert import convert_lightgbm
import onnxruntime as rt
import pandas as pd
from skl2onnx.common.data_types import FloatTensorType
import torch


def save_lgbm_model(x_train_df: pd.DataFrame, model, onnx_file_path: str):
    initial_type = [('float_input', FloatTensorType([None, x_train_df.shape[1]]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=8)
    onnx.checker.check_model(onnx_model)
    with open(onnx_file_path, "wb") as file_stream:
        file_stream.write(onnx_model.SerializeToString())


def save_torch_model(df: pd.DataFrame, model, onnx_file_path: str):
    torch.onnx.export(model,               # model being run
                      df,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


def predict_model(x_df, onnx_file_path: str):
    """
    Loads onnx model and predict
    @param x_df: Dataframe with feature for prediction
    @param onnx_file_path: Path to onnx-file with trained model
    """
    onnx_model_pred_test = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model_pred_test)
    sess = rt.InferenceSession(onnx_file_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    return sess.run([label_name], {input_name: x_df.astype(numpy.float32).to_numpy()})[0]
