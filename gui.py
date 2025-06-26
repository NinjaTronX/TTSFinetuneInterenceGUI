import gradio as gr
from TTS.api import TTS
from train_utils import fine_tune_model
import shutil, os, json, zipfile

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def synthesize(text, ref_audio=None):
    out = "output.wav"
    if ref_audio:
        tts.tts_with_vc_to_file(text=text, file_path=out, speaker_wav=ref_audio)
    else:
        tts.tts_to_file(text=text, file_path=out)
    return out

def upload_model(pth_file, config_file):
    global tts
    tts = TTS(model_path=pth_file, config_path=config_file)
    return "Custom model loaded."

def prepare_dataset(wavs):
    data_path = "/app/data/src/wavs"
    os.makedirs(data_path, exist_ok=True)
    for wav in wavs:
        shutil.copy(wav, data_path)
    return f"{len(wavs)} wavs uploaded."

def generate_metadata(transcript_texts):
    lines = transcript_texts.strip().split('\n')
    with open("/app/data/src/metadata.csv", "w") as f:
        for idx, line in enumerate(lines):
            audio_file = f"{idx}.wav"
            f.write(f"{audio_file}|{line.strip()}\n")
    return "metadata.csv created."

def extract_embeddings(wav_file):
    embed = tts.get_speaker_embedding(wav_file)
    path = "/app/embedding.json"
    with open(path, "w") as f:
        json.dump(embed.tolist(), f)
    return path

def train_model(zip_file, epochs):
    return fine_tune_model(zip_file, epochs)

def export_model():
    out_zip = "/app/exported_model.zip"
    base = "/app/fine_tuned"
    with zipfile.ZipFile(out_zip, 'w') as zipf:
        for root, _, files in os.walk(base):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), base))
    return out_zip

app = gr.Blocks()

with app:
    gr.Markdown("## 🎙 Coqui TTS Studio")

    with gr.Tab("Inference"):
        text = gr.Textbox(label="Text")
        ref = gr.Audio(label="Reference (optional)", type="filepath")
        out = gr.Audio()
        gr.Button("Synthesize").click(fn=synthesize, inputs=[text, ref], outputs=out)

    with gr.Tab("Custom Model"):
        model_file = gr.File(label=".pth", type="filepath")
        config_file = gr.File(label="config.json", type="filepath")
        msg = gr.Textbox()
        gr.Button("Load Model").click(fn=upload_model, inputs=[model_file, config_file], outputs=msg)

    with gr.Tab("Upload Dataset"):
        wavs = gr.File(file_types=[".wav"], file_count="multiple", type="filepath")
        upload_status = gr.Textbox()
        gr.Button("Upload WAVs").click(fn=prepare_dataset, inputs=wavs, outputs=upload_status)

        gr.Markdown("### Add Transcripts")
        transcripts = gr.Textbox(lines=10, label="Transcripts (one per line)")
        meta_status = gr.Textbox()
        gr.Button("Generate metadata.csv").click(fn=generate_metadata, inputs=transcripts, outputs=meta_status)

    with gr.Tab("Fine-Tune"):
        zip_file = gr.File(label="Training ZIP", type="filepath")
        epochs = gr.Slider(1, 50, 5)
        status = gr.Textbox()
        gr.Button("Start Training").click(fn=train_model, inputs=[zip_file, epochs], outputs=status)

    with gr.Tab("Export Model"):
        export_btn = gr.Button("Zip Fine-Tuned Model")
        zip_output = gr.File()
        export_btn.click(fn=export_model, outputs=zip_output)

    with gr.Tab("Extract Embedding"):
        emb_audio = gr.Audio(label="Speaker WAV", type="filepath")
        emb_file = gr.File()
        gr.Button("Extract").click(fn=extract_embeddings, inputs=emb_audio, outputs=emb_file)

app.launch(server_name="0.0.0.0", server_port=7860)
