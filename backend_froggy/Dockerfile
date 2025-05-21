FROM python:3.10-slim

ENV PORT=7860
ENV TOGETHER_API_KEY=placeholder-api-key  # Will be replaced by secret

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]