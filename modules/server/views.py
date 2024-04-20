from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from PIL import Image
import io
import os

from .db import Session

from modules.ml.predict import predict_captcha
from modules.ml.train import load_index_to_char, carregar_modelo


modelo_carregado = carregar_modelo()
index_to_char = load_index_to_char()


app = FastAPI(title='Image Captcha Api')


@app.post("/api/img_captcha")
async def create_captcha(image: UploadFile = File(...)):
    path = os.path.join('temp', 'gambiarra.png')
    with open(path, 'wb') as f:
        f.write(image.file.read())
        
    predicted_solution = predict_captcha(modelo_carregado, path, index_to_char)
    return JSONResponse(content={"solution": predicted_solution})
