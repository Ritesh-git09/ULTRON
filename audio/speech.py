
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from kokoro import KPipeline

from config import TTS_LANGUAGE, TTS_VOICE, TTS_SPEED
from audio.voice_fx import process_voice


class SpeechEngine:
    def wait_until_done(self):
    # Wait until all text has been synthesized.
        self.queue.join()

    # Then wait until all generated audio has played.
        self.audio_queue.join()

    def __init__(self):
        self.queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.pipeline = None
        self.worker = None
        self.playback_worker = None

        self.enabled = False
        self.first_audio_time = None
        self.turn_start = None

    def start(self):

        print("[ULTRON AUDIO] Loading Kokoro...")

        try:
            self.pipeline = KPipeline(
                lang_code=TTS_LANGUAGE
            )

            self.enabled = True

            # Start playback independently from synthesis.
            self.playback_worker = threading.Thread(
                target=self._play_worker,
                daemon=True
            )

            self.worker = threading.Thread(
                target=self._synth_worker,
                daemon=True
            )

            self.playback_worker.start()
            self.worker.start()

            print("[ULTRON AUDIO] Metallic voice online.")

        except Exception as error:
            self.enabled = False
            print(f"[TTS ERROR] {error}")
            print("[SYSTEM] Continuing in text-only mode.")

    def begin_turn(self):
        self.turn_start = time.perf_counter()
        self.first_audio_time = None

    def speak(self, text):

        if self.enabled and text.strip():
            self.queue.put(text.strip())

    def _synth_worker(self):

        while True:

            text = self.queue.get()

            try:

                if text is None:
                    self.audio_queue.put(None)
                    return

                generator = self.pipeline(
                    text,
                    voice=TTS_VOICE,
                    speed=TTS_SPEED
                )

                chunks = []

                for _, _, audio in generator:

                    if audio is None:
                        continue

                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()

                    chunks.append(
                        np.asarray(audio, dtype=np.float32)
                    )

                if not chunks:
                    continue

                original = np.concatenate(chunks)

                # Apply metallic effects once to the complete segment.
                processed = process_voice(original)

                # Hand audio to the playback worker immediately.
                self.audio_queue.put(processed)

            except Exception as error:
                print(f"\n[AUDIO SYNTH ERROR] {error}")

            finally:
                self.queue.task_done()

    def _play_worker(self):

        while True:

            audio = self.audio_queue.get()

            try:

                if audio is None:
                    return

                if self.first_audio_time is None:

                    self.first_audio_time = time.perf_counter()

                    if self.turn_start is not None:
                        latency = (
                            self.first_audio_time -
                            self.turn_start
                        )

                        print(
                            f"\n[AUDIO] First playback: "
                            f"{latency:.2f}s"
                        )

                sd.play(
                    audio,
                    samplerate=24000
                )

                sd.wait()

            except Exception as error:
                print(f"\n[AUDIO PLAYBACK ERROR] {error}")

            finally:
                self.audio_queue.task_done()

    def finish(self):

        if self.worker and self.worker.is_alive():

            self.queue.put(None)
            self.worker.join()

        if (
            self.playback_worker
            and self.playback_worker.is_alive()
        ):

            self.playback_worker.join()
