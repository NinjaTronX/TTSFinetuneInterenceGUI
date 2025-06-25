FROM python:3.10-slim

RUN apt-get update && apt-get install -y git ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install TTS gradio torch torchaudio

COPY gui.py /app/gui.py
COPY train_utils.py /app/train_utils.py
WORKDIR /app

EXPOSE 7860 6006

CMD ["python", "gui.py"]
