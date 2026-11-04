
from audio.listener import Listener

listener = Listener()

listener.start()

print("\n================================")
print("       ULTRON LISTENER TEST")
print("================================")
print("Speak naturally. Say 'exit' to stop.\n")

try:

    while True:

        text = listener.listen()

        if not text:
            continue

        if text.lower().strip() in (
            "exit",
            "quit",
            "stop listening"
        ):
            print("\n[ULTRON] Listener shutting down.")
            break

except KeyboardInterrupt:

    print("\n[ULTRON] Listener interrupted.")
