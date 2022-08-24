# FROM python:3.8-slim

# RUN pip install fastapi
# RUN pip install "uvicorn[standard]"

FROM python:3.8-slim

COPY . /app

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt


EXPOSE 8000