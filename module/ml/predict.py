import tensorflow as tf
import numpy as np
from typing import Optional

from .config import ALTURA, LARGURA


def decode_batch_predictions(
        pred: np.ndarray,
        num_to_char: tf.keras.layers.StringLookup,
        max_length: int
        ):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, : max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def predict_captcha(
        model: tf.keras.models.Model,
        img_path: str,
        num_to_char: tf.keras.layers.StringLookup,
        max_length: int,
        solution: Optional[str | None] = None,
        img_height: int = ALTURA,
        img_width: int = LARGURA
        ):
    # Carregar a imagem do caminho fornecido
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)  # Adicionando dimensão de lote
    
    # Fazer previsão
    pred = model.predict(img)
    
    # Decodificar a previsão
    pred_text = decode_batch_predictions(pred, num_to_char, max_length)[0]
    
    # Calcula a confiança
    confidences = []
    for prediction in pred[0]:
        max_prob_index = np.argmax(prediction)
        confidence = prediction[max_prob_index]
        confidences.append(confidence)
    mean_confidence = np.mean(confidences)
    
    # Calcula a precisão, se a solução for fornecida
    accuracy = None
    if solution:
        correct_chars = sum(1 for pred_char, real_char in zip(pred_text, solution) if pred_char == real_char)
        accuracy = correct_chars / len(solution)
    
    print(f'Predicted solution: {pred_text}')
    print(f'Real solution: {solution}')
    print(f'Accuracy: {accuracy}')
    print(f'Confidence: {mean_confidence}')
    
    return {
        "solution": pred_text,
        "confidence": float(mean_confidence),
        "accuracy": accuracy
    }