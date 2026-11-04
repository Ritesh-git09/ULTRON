
import time
from collections import deque

import numpy as np
import sounddevice as sd

from scipy.signal import resample_poly
from faster_whisper import WhisperModel

from config import (
    MIC_DEVICE,
    MIC_SAMPLE_RATE,
    MIC_CHANNELS,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)


class Listener:

    def __init__(self):

        self.model = None

        self.sample_rate = MIC_SAMPLE_RATE
        self.channels = MIC_CHANNELS
        self.device = MIC_DEVICE

        self.block_size = 1024
        self.silence_duration = 1.5
        self.max_recording_duration = 45

    def start(self):

        print("[ULTRON LISTENER] Loading Faster-Whisper...")

        try:

            self.model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )

            print("[ULTRON LISTENER] Speech recognition online.")

        except Exception as error:

            print(f"[LISTENER ERROR] {error}")
            raise

    def listen(self):

        if self.model is None:
            raise RuntimeError("Listener has not been started.")

        print("\n[ULTRON LISTENER] Listening...")

        pre_roll = deque(maxlen=10)

        recording = []
        speaking = False

        silence_frames = 0
        speech_frames = 0

        required_silence = int(
            self.silence_duration *
            self.sample_rate /
            self.block_size
        )

        max_frames = int(
            self.max_recording_duration *
            self.sample_rate /
            self.block_size
        )

        # Give the microphone a short ambient-noise calibration.
        noise_levels = []

        try:

            with sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.block_size
            ) as stream:

                for _ in range(20):

                    frame, overflowed = stream.read(
                        self.block_size
                    )

                    frame = frame[:, 0]

                    noise_levels.append(
                        np.sqrt(np.mean(frame ** 2))
                    )

                noise_floor = float(
                    np.median(noise_levels)
                )

                threshold = max(
                    0.003,
                    noise_floor * 2.0
                )

                print("[ULTRON LISTENER] Ready. Speak now.")

                start_time = time.perf_counter()

                while True:

                    frame, overflowed = stream.read(
                        self.block_size
                    )

                    frame = frame[:, 0].copy()

                    volume = np.sqrt(
                        np.mean(frame ** 2)
                    )

                    pre_roll.append(frame)

                    if not speaking:

                        if volume > threshold:

                            speaking = True

                            recording.extend(pre_roll)

                            speech_frames = 1

                            print(
                                "[ULTRON LISTENER] "
                                "Speech detected..."
                            )

                    else:

                        recording.append(frame)

                        if volume > threshold * 0.60:

                            silence_frames = 0
                            speech_frames += 1

                        else:

                            silence_frames += 1

                    if speaking:

                        if speech_frames > max_frames:
                            break

                        if silence_frames >= required_silence:
                            break

                    # Avoid waiting forever for speech.
                    if not speaking and (
                        time.perf_counter() - start_time > 30
                    ):
                        print(
                            "[ULTRON LISTENER] "
                            "No speech detected."
                        )
                        return ""

        except Exception as error:

            print(f"[MICROPHONE ERROR] {error}")
            return ""

        if speech_frames < 5:

            print("[ULTRON LISTENER] Speech too short.")
            return ""

        audio = np.concatenate(recording)

        # Convert microphone audio from 44.1 kHz to 16 kHz.
        audio_16k = resample_poly(
            audio,
            160,
            441
        ).astype(np.float32)

        print("[ULTRON LISTENER] Transcribing...")

        try:

            segments, info = self.model.transcribe(
                audio_16k,
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300
                },
                condition_on_previous_text=False
            )

            transcript = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            if transcript:

                print(
                    f"\n[YOU SAID]: {transcript}"
                )

            else:

                print("[ULTRON LISTENER] No words recognized.")

            return transcript

        except Exception as error:

            print(f"[TRANSCRIPTION ERROR] {error}")
            return ""
