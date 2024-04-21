from module.ml.train import train


if __name__ == "__main__":
    train(epochs=25, batch_size=6, extensions=('.png'))