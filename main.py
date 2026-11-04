
import json
from pathlib import Path

from core.llm import UltronLLM
from audio.speech import SpeechEngine
from audio.listener import Listener

from config import ENABLE_TTS, HISTORY_FILE


def load_history(ultron):

    path = Path(HISTORY_FILE)

    if not path.exists():
        return

    try:
        saved = json.loads(
            path.read_text(encoding="utf-8")
        )

        for message in saved:

            if message.get("role") in ("user", "assistant"):
                ultron.messages.append(message)

        print("[MEMORY] Previous conversation restored.")

    except Exception as error:
        print(f"[MEMORY] Could not restore history: {error}")


def save_history(ultron):

    path = Path(HISTORY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    conversation = [
        message
        for message in ultron.messages
        if message["role"] in ("user", "assistant")
    ]

    path.write_text(
        json.dumps(
            conversation,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def main():

    print("""
========================================
                 ULTRON
         ARTIFICIAL INTELLIGENCE
========================================
    """)

    ultron = UltronLLM()

    load_history(ultron)

    speech = None
    listener = Listener()

    try:

        # Initialize speech output.
        if ENABLE_TTS:

            speech = SpeechEngine()
            speech.start()

        # Initialize microphone and recognition.
        listener.start()

        print("\n[SYSTEM] ULTRON ONLINE.")
        print("[SYSTEM] Voice interaction activated.")
        print("[SYSTEM] Say 'exit' to terminate.\n")

        while True:

            # Listen for the user's voice.
            user_input = listener.listen()

            if not user_input:
                continue

            command = user_input.lower().strip()

            if command in (
                "exit",
                "quit",
                "shut down",
                "shutdown ultron"
            ):

                print("\n[ULTRON] Shutting down.")
                break

            if command in (
                "clear conversation",
                "reset conversation"
            ):

                ultron.messages = [
                    ultron.messages[0]
                ]

                save_history(ultron)

                print("[SYSTEM] Conversation reset.\n")
                continue

            # Process the user's request.
            ultron.generate(
                user_input,
                speech=speech
            )

            # Wait until ULTRON finishes speaking.
            if speech and speech.enabled:
                speech.wait_until_done()

            save_history(ultron)

    except KeyboardInterrupt:

        print("\n[SYSTEM] Interrupt received.")

    finally:

        if speech:
            speech.finish()

        save_history(ultron)

        print("\nULTRON: Until next time.")


if __name__ == "__main__":
    main()
