

from modules.ml.predict import predict_captcha
from modules.ml.train import load_character_maps, carregar_modelo


modelo_carregado = carregar_modelo()
char_map = load_character_maps()
print(char_map['index_to_char'])


if __name__ == '__main__':
    captcha_image_path = r'samples\3fbxd.png'  
    predicted_solution = predict_captcha(modelo_carregado, captcha_image_path, char_map)
    print('Predicted solution:', predicted_solution)