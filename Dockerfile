FROM replicate/python:3.10-slim

RUN apt-get update && apt-get install -y git
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY predict.py /app/predict.py

CMD ["cog", "run", "python", "/app/predict.py"]
