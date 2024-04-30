FROM python:3.10.9-slim
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip cache purge
COPY . ./ml_captcha_image
USER root
WORKDIR /ml_captcha_image
EXPOSE 8000
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
