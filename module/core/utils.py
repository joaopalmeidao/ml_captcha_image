import random
import string


def random_generator(size: int=10, chars=string.ascii_lowercase + string.digits) -> str:
        """Gera uma sequencia randomica

        Args:
            size (int, optional): tamanho da string. Defaults to 10.
            chars (_type_, optional): caracteres aceitos na randomizacao. Defaults to string.ascii_lowercase+string.digits.

        Returns:
            str: sequencia randomica
        """
        return ''.join(random.choice(chars) for _ in range(size))
