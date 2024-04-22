import tensorflow as tf
import numpy as np
from typing import Optional

from .pre_process import pre_process_img_cv, pre_process_img


def predict_captcha(
    model: tf.keras.models.Sequential,
    captcha_image_path: str,
    index_to_char: dict,
    solution: Optional[str | None] =None,
    verbose: Optional[bool] = False
    ) -> dict:
    correct_chars = None
    accuracy = None
    mean_confidence = None
    
    img = pre_process_img(captcha_image_path)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    
    decoded_predictions = []
    confidences = []
    for prediction in predictions[0]:
        predicted_index = np.argmax(prediction)
        predicted_char = index_to_char[predicted_index]
        confidence = prediction[predicted_index]
        decoded_predictions.append(predicted_char)
        confidences.append(confidence)
    
    predicted_solution = ''.join(decoded_predictions)
    mean_confidence = np.mean(confidences)
    
    if solution:
        correct_chars = sum(1 for pred, real in zip(predicted_solution, solution) if pred == real)
        accuracy = correct_chars / len(solution)
    
    result = {
        "solution": predicted_solution,
        "accuracy": accuracy,
        "confidence": mean_confidence.astype(float)
    }
    
    if verbose:
        print(f'Predicted solution: {predicted_solution}')
        print(f'Real solution: {solution}')
        print(f'Accuracy: {accuracy}')
        print(f'Confidence: {mean_confidence}')
    
    return result