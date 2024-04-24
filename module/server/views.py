from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os

from .db import Session

from module.core.utils import random_generator
from module.ml.model import load_model
from module.ml.predict import predict_captcha


data_modelo = load_model()


app = FastAPI(title='Image Captcha Api')


@app.post("/api/img_captcha")
async def submit_img_captcha(
    image: UploadFile = File(..., media_type="image/*"),
    solution: Optional[str | None] = None
    ) -> JSONResponse:
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
            
        result = predict_captcha(
            data_modelo['model'],
            path,
            data_modelo['num_to_char'],
            max([len(label) for label in data_modelo['labels']]),
            solution,
            )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        try:
            os.remove(path)
        except:
            pass
