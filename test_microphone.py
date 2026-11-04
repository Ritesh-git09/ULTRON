
import sounddevice as sd

print("\n========== AVAILABLE MICROPHONES ==========\n")

devices = sd.query_devices()

for index, device in enumerate(devices):

    if device["max_input_channels"] > 0:
        print(
            f"Device ID: {index}\n"
            f"Name: {device['name']}\n"
            f"Input channels: {device['max_input_channels']}\n"
            f"Sample rate: {device['default_samplerate']}\n"
            "------------------------------------------"
        )

print("\nDefault input device:")
print(sd.default.device[0])
