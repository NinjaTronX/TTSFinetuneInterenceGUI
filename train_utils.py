# train_utils.py
import os
import zipfile
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.api import TTS

def fine_tune_model(zip_path, epochs):
    out = "/app/fine_tuned"
    if os.path.exists(out):
        return f"Model already exists at {out}"

    os.makedirs("/app/data/src", exist_ok=True)
    with zipfile.ZipFile(zip_path.name) as z:
        z.extractall("/app/data/src")

    # Load base config and update dataset path
    config = GlowTTSConfig(
        output_path=out,
        epochs=epochs,
        datasets=[BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.csv", path="/app/data/src")]
    )
    ap = AudioProcessor.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(config.datasets[0], eval_split=True)
    model = GlowTTS(config, ap, TTS)  # or speaker_manager=None

    trainer = Trainer(TrainerArgs(), config, out, model=model,
                      train_samples=train_samples,
                      eval_samples=eval_samples)
    trainer.fit()

    return f"Finished! Model saved at {out}"
