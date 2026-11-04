
import numpy as np
import sounddevice as sd

from kokoro import KPipeline
from audio.voice_fx import process_voice

pipeline = KPipeline(lang_code="a")

text = (
    "I have observed humanity for quite some time. "
    "Your ambitions are remarkable. "
    "Your methods, however, leave much to be desired. "
    "Let us begin."
)

print("[ULTRON] Generating original voice...")

generator = pipeline(
    text,
    voice="am_michael",
    speed=0.90
)

chunks = []

for _, _, audio in generator:

    if audio is None:
        continue

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    chunks.append(
        np.asarray(audio, dtype=np.float32).reshape(-1)
    )

original = np.concatenate(chunks)

print("\nPlaying ORIGINAL voice...")
sd.play(original, 24000)
sd.wait()

input("\nPress ENTER to hear ULTRON...")

print("\nApplying ULTRON effects...")

processed = process_voice(original)

print("Playing ULTRON voice...")
sd.play(processed, 24000)
sd.wait()

print("\nVoice audition completed.")
