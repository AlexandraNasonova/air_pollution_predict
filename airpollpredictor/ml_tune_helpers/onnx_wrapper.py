import onnx
from onnxmltools.convert import convert_lightgbm
import onnxruntime as rt
import pandas as pd
from skl2onnx.common.data_types import FloatTensorType
import numpy


def save_model(x_train_df: pd.DataFrame, model, onnx_file_path: str):
    initial_type = [('float_input', FloatTensorType([None, x_train_df.shape[1]]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=8)
    onnx.checker.check_model(onnx_model)
    with open(onnx_file_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def predict_model(x_df, onnx_file_path: str):
    onnx_model_pred_test = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model_pred_test)
    sess = rt.InferenceSession(onnx_file_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    return sess.run([label_name], {input_name: x_df.astype(numpy.float32).to_numpy()})[0]