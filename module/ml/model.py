import pickle
import tensorflow as tf
from typing import Optional

from .config import CAMINHO_DATA_MODEL

def save_model(
        model: tf.keras.models.Model,
        char_to_num: tf.keras.layers.StringLookup,
        num_to_char: tf.keras.layers.StringLookup,
        LABELS: list[str],
        filename: Optional[str] = CAMINHO_DATA_MODEL) -> None:
    # Salvar o modelo
    char_to_num_dict = char_to_num.get_vocabulary()
    num_to_char_dict = num_to_char.get_vocabulary()

    data = {
        'model': model,
        'labels': LABELS,
        'char_to_num': char_to_num_dict,
        'num_to_char': num_to_char_dict
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def load_model(filename: Optional[str] = CAMINHO_DATA_MODEL) -> dict:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    char_to_num = tf.keras.layers.StringLookup( 
        vocabulary=list(data['char_to_num']), mask_token=None
    ) 

    num_to_char = tf.keras.layers.StringLookup( 
        vocabulary=list(data['num_to_char']), 
        mask_token=None, invert=True
    ) 
    
    data['num_to_char'] = num_to_char
    data['char_to_num'] = char_to_num
    
    return data 