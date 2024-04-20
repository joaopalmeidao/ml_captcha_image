import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

from .config import ALTURA, LARGURA
from .pre_process import pre_process_img


def predict_captcha(model, captcha_image_path, index_to_char):
    img = pre_process_img(captcha_image_path)
    
    predictions = model.predict(img)
    
    decoded_predictions = []
    for prediction in predictions[0]:
        predicted_index = np.argmax(prediction)
        predicted_char = index_to_char[predicted_index]
        decoded_predictions.append(predicted_char)
    
    return ''.join(decoded_predictions)