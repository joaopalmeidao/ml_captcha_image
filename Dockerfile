FROM python:3.10.9-slim
RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser/ml_captcha_image
RUN chown -R appuser:appuser /home/appuser/ml_captcha_image
COPY . .
USER appuser
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
RUN chmod +x entrypoint.sh
ENTRYPOINT ["~/ml_captcha_image/entrypoint.sh"]
