import zipfile, os
from TTS.bin.finetune import FinetuneTrainer

def fine_tune_model(zip_path, epochs):
    base = "tts_models/en/ljspeech/tacotron2-DDC"
    target = "fine_tuned"
    out = "/app/" + target
    os.makedirs(out, exist_ok=True)

    with zipfile.ZipFile(zip_path.name) as z:
        z.extractall("/app/data")

    trainer = FinetuneTrainer(config_path=base + ".json",
                              model_path=base + ".pth",
                              dataset_config="/app/data/config.json",
                              output_path=out,
                              epochs=epochs)
    trainer.fine_tune()
    return f"Finished! Model saved at {out}"
