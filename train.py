from module.ml.train import train, CAMINHO_MODELO


if __name__ == "__main__":
    train(
        epochs=100,
        batch_size=16,
        extensions=('.png'),
        # existing_model_path=CAMINHO_MODELO
        )