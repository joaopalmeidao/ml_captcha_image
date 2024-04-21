import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
from typing import Optional, Iterable

from .config import ALTURA, LARGURA
from .pre_process import pre_process_img


CAMINHO_MODELO: str = os.path.join('models','epochs_100_seq2seq.keras')
CAMINHO_INDEX_TO_CHAR: str = os.path.join('models','index_to_char.pickle')

def carregar_dataset(
    directory: Optional[str] = 'samples',
    extensions: Iterable[str] = ('.png', '.jpg'), 
    max_size: int | None = None
    ) -> pd.DataFrame:
    
    lista_imagens = [i for i in os.listdir(directory) if i.endswith(extensions)]
    
    if max_size:
        lista_imagens = lista_imagens[:max_size]
        
    df = pd.DataFrame({'imagens': lista_imagens})
    df['solucao'] = df['imagens'].apply(lambda x: os.path.splitext(x)[0])
    df['caminho_imagem'] = df['imagens'].apply(lambda x: os.path.join(directory, x))
    return df

def load_images(file_paths: Iterable[str], altura: Optional[int] = ALTURA, largura: Optional[int] = LARGURA) -> np.array:
    images = []
    for path in file_paths:
        img = pre_process_img(path, altura=altura, largura=largura, train=True)
        images.append(img)
    return np.array(images)
    
def salvar_modelo(model: models.Sequential, path: Optional[str] = CAMINHO_MODELO) -> None:
    model.save(path)

def carregar_modelo(path: Optional[str] = CAMINHO_MODELO) -> models.Sequential:
    return load_model(path)

def salvar_index_to_char(index_to_char: dict, path: Optional[str] = CAMINHO_INDEX_TO_CHAR) -> None:
    with open(path, 'wb') as f:
        pickle.dump(index_to_char, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_index_to_char(path: Optional[str] = CAMINHO_INDEX_TO_CHAR) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def train(epochs: int = 100, batch_size: int = 128) -> None: 
    df = carregar_dataset()
    print(df)
    
    X = load_images(df['caminho_imagem'])

    y = df['solucao'].apply(list)

    characters = set(char for sublist in y for char in sublist)
    num_classes = len(characters)

    char_to_index = {char: i for i, char in enumerate(characters)}
    index_to_char = {i: char for char, i in char_to_index.items()}
    
    y = [[char_to_index[char] for char in sublist] for sublist in y]

    max_length = max(len(sublist) for sublist in y)
    y = tf.keras.preprocessing.sequence.pad_sequences(
        y, maxlen=max_length, padding='post'
        )

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ALTURA, LARGURA, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.RepeatVector(max_length),
        layers.LSTM(64, return_sequences=True),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    
    salvar_index_to_char(index_to_char)
    salvar_modelo(model)
