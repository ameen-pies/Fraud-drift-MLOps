FROM python:3.11-slim
WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
COPY training/requirements.txt ./training/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt -r training/requirements.txt

COPY api/ ./api/
COPY training/ ./training/
COPY data/ ./data/
COPY tests/ ./tests/

RUN chmod +x api/entrypoint.sh

CMD ["sh", "api/entrypoint.sh"]
