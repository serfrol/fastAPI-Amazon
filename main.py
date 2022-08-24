from typing import Optional, Callable
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, Request, Response, HTTPException, Form, UploadFile, File, Depends, Query
from fastapi.responses import FileResponse
import numpy as np
from sklearn import datasets
from joblib import dump, load
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
import json
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import io
# from schemas import BaseForm
import pandas as pd
# import aiofiles
from pathlib import Path
import shutil
from fastapi.responses import HTMLResponse


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")

rf_h = load('rf_h.pkl')
rf_l = load('rf_l.pkl')
man_list = list(load('man_list.pkl'))

def fill_with_ones(row):
  row[f'category_{row.category}'] = 1
  row[f'sub_category_{row.sub_category}'] = 1
  row[f'manufacturer_{row.manufacturer}'] = 1
  return row

def get_predicts(df):
    df['manufacturer_check'] = 0
    df.loc[df.manufacturer.isin(man_list), 'manufacturer_check'] = 1
    cat_cols = list(df.select_dtypes(include=['object']).columns.to_list())

    d = dict.fromkeys(rf_l.feature_names_in_[10:], 0)
    temp_df = pd.DataFrame(d, index=df.index)
    df = pd.concat([df, temp_df], axis=1)
    df = df.apply(fill_with_ones, axis=1)
    
    df_h = df[df['manufacturer_check'] == 1].drop(cat_cols, axis=1).copy()
    df_l = df[df['manufacturer_check'] == 0].drop(cat_cols, axis=1).copy()
    
    h_pred = rf_h.predict(df.drop(cat_cols, axis=1)[df.index.isin(df_h.index)])
    l_pred = rf_l.predict(df.drop(cat_cols, axis=1)[df.index.isin(df_l.index)])

    pred_df = pd.DataFrame()
    pred_df['Prediction'] = list(h_pred) + list(l_pred)
    pred_df.index = list(df_h.index) + list(df_l.index)
    pred_df = pred_df.sort_index()

    return pred_df

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('upload.html', {"request": request})

@app.post('/', response_class=HTMLResponse)
def get_file(filename: str = Form(...), file: UploadFile = File(...)):

    filepath = f"files/{filename}.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(filepath)
    pred_df = get_predicts(df)
    pred_df.to_csv(f'files/{filename}_pred.csv')
    new_filepath = f'files/{filename}_pred.csv'
    
    return FileResponse(new_filepath, media_type='application/octet-stream',filename=f'{filename}_pred.csv')

