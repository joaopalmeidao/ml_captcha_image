FROM python:3.10.9-slim
RUN apt-get update
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./ml_captcha_image  
USER root
WORKDIR /ml_captcha_image  
EXPOSE 8000
RUN chown -R root:root /ml_captcha_image  
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]