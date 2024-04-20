from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from datetime import datetime
from PIL import Image
import io
import os

from .db import Session

from module.ml.predict import predict_captcha
from module.ml.train import load_index_to_char, carregar_modelo
from module.core.utils import random_generator


modelo_carregado = carregar_modelo()
index_to_char = load_index_to_char()


app = FastAPI(title='Image Captcha Api')


@app.post("/api/img_captcha")
async def submit_img_captcha(image: UploadFile = File(...)):
    try:
        foder = 'temp'
        if not os.path.exists(foder):
            os.makedirs(foder)
            
        path = os.path.join(
            foder,
            f"img_{random_generator()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.png"
            )
        
        with open(path, 'wb') as f:
            f.write(image.file.read())
            
        predicted_solution = predict_captcha(modelo_carregado, path, index_to_char)
        return JSONResponse(content={"solution": predicted_solution})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(path)
        except:
            pass
