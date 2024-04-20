import random
import string
from typing import Optional


def random_generator(
    size: Optional[int] = 10, 
    chars: Optional[str] = string.ascii_lowercase + string.digits
    ) -> str:
        """Gera uma sequencia randomica

        Args:
            size (int, optional): tamanho da string. Defaults to 10.
            chars (str, optional): caracteres aceitos na randomizacao. Defaults to string.ascii_lowercase+string.digits.

        Returns:
            str: sequencia randomica
        """
        return ''.join(random.choice(chars) for _ in range(size))
