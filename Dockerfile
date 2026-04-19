FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./
COPY prepare.py ../prepare.py
COPY dashboard/ ./static/

EXPOSE 8080
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
