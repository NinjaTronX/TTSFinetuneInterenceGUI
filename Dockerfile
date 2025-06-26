FROM python:3.10-slim

# Add essential build tools
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install git+https://github.com/coqui-ai/TTS#egg=TTS[trainers]
RUN pip install gradio torch torchaudio


COPY gui.py /app/gui.py
COPY train_utils.py /app/train_utils.py
WORKDIR /app

EXPOSE 7860 6006 22

CMD ["python", "gui.py"]
