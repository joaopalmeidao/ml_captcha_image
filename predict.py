

from module.ml.predict import predict_captcha
from module.ml.train import load_index_to_char, carregar_modelo


modelo_carregado = carregar_modelo()
index_to_char = load_index_to_char()


if __name__ == '__main__':
    captcha_image_path = r'samples\3fbxd.png'  
    predicted_solution = predict_captcha(modelo_carregado, captcha_image_path, index_to_char)
    print('Predicted solution:', predicted_solution)