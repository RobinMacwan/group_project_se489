FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY se489_group_project/ se489_group_project/
COPY models/ models/
COPY templates/ templates/
COPY app.py/ app.py/

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]