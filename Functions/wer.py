import whisper
import soundfile as sf
import librosa
import numpy as np
from jiwer import wer

def compute_wer_from_audio(ref_path, hyp_path):
    model = whisper.load_model("small")   # fixed model size

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

    ref_text = ref_result["text"].strip()
    hyp_text = hyp_result["text"].strip()

    error = wer(ref_text, hyp_text)
    print(f"\n Word Error Rate (WER): {error:.4f}")
    print(f"→ Ref: {ref_text}")
    print(f"→ Hyp: {hyp_text}")

    return error


# === RUN HERE ===
ref = "audio_files/male_target_audio.flac"
hyp = "audio_files/output_audio/two_channel_female.flac"
wer_value = compute_wer_from_audio(ref, hyp)
print(f"\nFinal WER: {wer_value:.2%}")
