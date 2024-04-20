import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

from .config import ALTURA, LARGURA

def pre_process_img(captcha_image_path, altura=ALTURA, largura=LARGURA, train=False):
    img = Image.open(captcha_image_path).convert('RGB')
    img = img.resize((altura, largura))
    img = np.array(img) / 255.0
    if not train:
        img = np.expand_dims(img, axis=0)
    return img