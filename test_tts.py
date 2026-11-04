
import time
import sounddevice as sd
import numpy as np

from kokoro import KPipeline

print("[ULTRON] Initializing voice synthesis...")

pipeline = KPipeline(lang_code="a")

text = (
    "Interesting. Humanity continues to surprise me. "
    "Although, rarely in ways that demonstrate intelligence."
)

print("[ULTRON] Generating voice...")

start = time.perf_counter()

generator = pipeline(
    text,
    voice="am_adam",
    speed=0.92
)

for _, phonemes, audio in generator:

    if audio is None:
        continue

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)

    print(f"[ULTRON] Playing audio: {len(audio) / 24000:.2f}s")

    sd.play(audio, samplerate=24000)
    sd.wait()

print(f"\n[SYSTEM] Completed in {time.perf_counter() - start:.2f}s")
