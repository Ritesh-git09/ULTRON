
import re
import time

from ollama import Client

from config import MODEL, OLLAMA_HOST, KEEP_ALIVE
from core.personality import load_personality


class UltronLLM:

    def __init__(self):

        self.client = Client(host=OLLAMA_HOST)

        self.messages = [
            {
                "role": "system",
                "content": load_personality()
            }
        ]

    def generate(self, user_input, speech=None):

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        response_text = ""
        pending_text = ""

        start = time.perf_counter()
        first_token = None

        if speech and speech.enabled:
            speech.begin_turn()

        print("\nULTRON: ", end="", flush=True)

        try:

            stream = self.client.chat(
                model=MODEL,
                messages=self.messages,
                stream=True,
                keep_alive=KEEP_ALIVE,
                options={
                    "temperature": 0.8,
                    "top_p": 0.9,
                }
            )

            for chunk in stream:

                token = chunk.message.content or ""

                if not token:
                    continue

                if first_token is None:
                    first_token = time.perf_counter()

                print(token, end="", flush=True)

                response_text += token
                pending_text += token

                # Send completed sentences to TTS.
                while True:

                    match = re.search(
                        r'[.!?](?=\s|$)',
                        pending_text
                    )

                    if not match:
                        break

                    sentence = pending_text[:match.end()].strip()

                    pending_text = pending_text[match.end():]

                    if sentence and speech and speech.enabled:
                        speech.speak(sentence)

                # Avoid excessively long audio buffers.
                if len(pending_text) > 180:

                    split_at = pending_text.rfind(" ", 0, 180)

                    if split_at > 40:

                        phrase = pending_text[:split_at].strip()
                        pending_text = pending_text[split_at:].strip()

                        if speech and speech.enabled:
                            speech.speak(phrase)

            # Send remaining text.
            if pending_text.strip() and speech and speech.enabled:
                speech.speak(pending_text.strip())

            self.messages.append({
                "role": "assistant",
                "content": response_text
            })

            end = time.perf_counter()

            print("\n\n--- ULTRON PERFORMANCE ---")

            if first_token:
                print(
                    f"First token latency: "
                    f"{first_token - start:.2f}s"
                )

            print(
                f"Generation time: {end - start:.2f}s"
            )

            print(
                f"Response length: {len(response_text)} characters"
            )

            print()

            return response_text

        except Exception:

            self.messages.pop()
            raise
