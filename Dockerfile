FROM python:3.10-slim

WORKDIR /app

RUN pip install pydantic openai pyyaml

COPY . .

CMD ["python", "inference.py"]
