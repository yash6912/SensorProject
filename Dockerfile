FROM python:3.8-slim-buster

WORKDIR /app

COPY  . /app/

RUN pip install -r requiremennts.txt

CMD ["python#","app.py"]