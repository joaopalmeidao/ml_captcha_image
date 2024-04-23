import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, Literal

from .config import ALTURA, LARGURA


def pre_process_img(
    captcha_image_path: str,
    altura: Optional[int] = ALTURA,
    largura: Optional[int] = LARGURA,
    img_mode: Literal['L', 'RGB'] = 'L'
    ) -> np.array:
    img = tf.io.read_file(captcha_image_path) 
    # Converting the image to grayscale 
    img = tf.io.decode_png(img, channels=1) 
    img = tf.image.convert_image_dtype(img, tf.float32) 
    # Resizing to the desired size 
    img = tf.image.resize(img, [altura, largura]) 
    # Transposing the image 
    img = tf.transpose(img, perm=[1, 0, 2]) 
    
    return img

def pre_process_img_cv(
    captcha_image_path: str,
    altura: Optional[int] = ALTURA,
    largura: Optional[int] = LARGURA,
    ) -> np.array:
    
    img = cv.imread(captcha_image_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (altura, largura))
    
    # Limiar adaptativo
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 17, 2)
    
    # Limiar de Otsu
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Suavização gaussiana e limiar de Otsu
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Dilatação
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(th, kernel, iterations=1)
    dilation2 = cv.dilate(th2, kernel, iterations=1)
    dilation3 = cv.dilate(th3, kernel, iterations=1)
    
    # Erosão
    erosion = cv.erode(dilation, kernel, iterations=1)
    erosion2 = cv.erode(dilation2, kernel, iterations=1)
    erosion3 = cv.erode(dilation3, kernel, iterations=1)
    
    # Dilatação após a erosão
    dilation_after_erosion = cv.dilate(erosion, kernel, iterations=1)
    dilation2_after_erosion = cv.dilate(erosion2, kernel, iterations=1)
    dilation3_after_erosion = cv.dilate(erosion3, kernel, iterations=1)
    
    # Concatenação de todas as imagens processadas
    processed_images = np.array([img, th, th2, th3, erosion, erosion2, erosion3, dilation_after_erosion, dilation2_after_erosion, dilation3_after_erosion])
    
    img = np.array(dilation3_after_erosion) / 255.0
    
    return img
