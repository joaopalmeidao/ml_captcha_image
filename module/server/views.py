from fastapi import HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os
import json
import importlib.metadata
import sys

from .db import Session
from .app import app

from module.core.utils import random_generator
from module.ml.model import load_model
from module.ml.predict import predict_captcha
from module import __version__


model_data: dict = load_model()


@app.get("/api/model/get_config", 
        tags=["Model"],
        summary="Get model configuration",
        description="Retrieve the configuration of the image captcha recognition model."
        )
async def get_config() -> JSONResponse:
    try:
        result = model_data['model'].to_json()
        return JSONResponse(content=json.loads(result))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/captcha/img_captcha",
        tags=["Captcha"],
        summary="Submit image captcha",
        description="Submit an image captcha to get the solution."
        )
async def submit_img_captcha(
    image: UploadFile = File(..., media_type="image/*"),
    solution: Optional[str | None] = None,
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
            model_data['model'],
            path,
            model_data['num_to_char'],
            max([len(label) for label in model_data['labels']]),
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

@app.get("/api/utility/package_versions", 
        tags=["Utility"],
        summary="Get package versions",
        description="Retrieve the versions of packages used in the project."
)
async def get_package_versions() -> JSONResponse:
    try:
        package_versions = {}
        for distribution in importlib.metadata.distributions():
            package_versions[distribution.metadata["Name"]] = distribution.version
        
        return JSONResponse(content=package_versions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/utility/python_version", 
        tags=["Utility"],
        summary="Get Python version",
        description="Retrieve the version of Python used in the project."
)
async def get_python_version() -> str:
    try:
        python_version = sys.version
        return python_version
    
    except Exception as e:
        return str(e)