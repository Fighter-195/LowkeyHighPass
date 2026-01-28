# !apt-get install -y ffmpeg
# !pip install -q git+https://github.com/openai/whisper.git
# !pip install -q soundfile librosa jiwer

import whisper
import soundfile as sf
import librosa
import numpy as np
from jiwer import wer
import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_wer_from_audio(ref_path, hyp_path):
    model = whisper.load_model("small") 

    def load_and_preprocess(path):
        audio, sr = sf.read(path)
        if sr != 16000:
            audio = librosa.resample(np.asarray(audio, dtype=np.float32),
                                     orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio.astype(np.float32)

    ref_audio = load_and_preprocess(ref_path)
    hyp_audio = load_and_preprocess(hyp_path)

    print(f"🎧 Transcribing reference: {ref_path}")
    ref_result = model.transcribe(ref_audio, fp16=False)
    
    print(f"🎧 Transcribing hypothesis: {hyp_path}")
    hyp_result = model.transcribe(hyp_audio, fp16=False)

    ref_text_raw = ref_result["text"].strip()
    hyp_text_raw = hyp_result["text"].strip()
    
    ref_text = normalize_text(ref_text_raw)
    hyp_text = normalize_text(hyp_text_raw)

    error = wer(ref_text, hyp_text)
    
    print(f"\n📝 Normalized transcriptions:")
    print(f"→ Ref: {ref_text}")
    print(f"→ Hyp: {hyp_text}")
    print(f"\n📊 Word Error Rate (WER): {error:.4f}")

    return error


# === RUN HERE ===
ref = "target_11.flac"
hyp = "processed_signal.wav"
wer_value = compute_wer_from_audio(ref, hyp)
