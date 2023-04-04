from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
from mini_ml import model
from data_processor import data_proc

app = FastAPI()

@app.post("/predict_aqi", response_class=FileResponse)
async def predict_aqi(file: UploadFile = File(...)):
    data = pd.read_csv(file.file, index_col=0)
    df = data_proc(data)
    y = model.predict(df)
    data['predicted_aqi'] = y
    data.to_csv('predictions.csv')
    return FileResponse('predictions.csv')