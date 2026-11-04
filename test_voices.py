
import sounddevice as sd
import numpy as np
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")

voices = [
    "am_onyx",
    "am_fenrir",
    "am_michael",
    "am_echo",
]

text = (
    "I have observed humanity for quite some time. "
    "Your ambitions are remarkable. "
    "Your methods, however, leave much to be desired. "
    "Let us begin."
)

for voice in voices:

    print(f"\n[ULTRON VOICE TEST] {voice}")

    generator = pipeline(
        text,
        voice=voice,
        speed=0.88
    )

    for _, _, audio in generator:

        if audio is None:
            continue

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32)

        sd.play(audio, samplerate=24000)
        sd.wait()

    input("Press ENTER to hear the next voice...")

print("Voice audition completed.")
