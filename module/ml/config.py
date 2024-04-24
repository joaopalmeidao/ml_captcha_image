import os
from typing import Iterable

ALTURA: int = 50
LARGURA: int = 200

BATCH_SIZE: int = 16

MAX_SIZE_DATASET: int | None = 2000

EXTENSIONS: Iterable[str]  = ('png', 'jpg', 'jpeg')

CAMINHO_DATA_MODEL: str = os.path.join('models','model1.pkl')