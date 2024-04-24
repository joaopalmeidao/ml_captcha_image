from fastapi import FastAPI

from module import __version__


app = FastAPI(title='Image Captcha Api', version=__version__)

app.description = """
This is an application for Image Captcha Recognition.
"""

app.contact = {
    "name": "João Pedro Almeida Oliveira",
    "url": "https://www.linkedin.com/in/joaopalmeidao/",
    "email": "",
}

app.license_info = {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
}