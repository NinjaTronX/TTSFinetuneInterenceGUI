import gradio as gr
from TTS.api import TTS
from train_utils import fine_tune_model

# Initialize inference model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

def synthesize(text):
    outfile = "output.wav"
    tts.tts_to_file(text=text, file_path=outfile)
    return outfile

def train(uploaded_zip, epochs):
    return fine_tune_model(uploaded_zip, epochs)

app = gr.Blocks()

with app:
    gr.Markdown("# Coqui TTS — Inference & Fine‑Tuning")
    with gr.Tab("Inference"):
        text_in = gr.Textbox(label="Text for synthesis")
        out = gr.Audio()
        gr.Button("Synthesize").click(fn=synthesize, inputs=text_in, outputs=out)
    with gr.Tab("Fine‑Tune"):
        data = gr.File(label="Upload training data ZIP", file_count="single", type="file")
        epochs = gr.Slider(1, 50, value=5, step=1, label="Epochs")
        status = gr.Textbox(label="Training status")
        gr.Button("Start Fine‑Tune").click(fn=train, inputs=[data, epochs], outputs=status)

app.launch(server_name="0.0.0.0", server_port=7860)
