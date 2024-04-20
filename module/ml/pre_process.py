import numpy as np
from PIL import Image
from typing import Optional

from .config import ALTURA, LARGURA


def pre_process_img(
    captcha_image_path: str,
    altura: Optional[int] = ALTURA,
    largura: Optional[int] = LARGURA,
    train: Optional[bool] = False
    ) -> np.array:
    img = Image.open(captcha_image_path).convert('RGB')
    img = img.resize((altura, largura))
    img = np.array(img) / 255.0
    if not train:
        img = np.expand_dims(img, axis=0)
    return img