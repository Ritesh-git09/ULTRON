MODEL = "llama3.1:latest"

OLLAMA_HOST = "http://localhost:11434"
KEEP_ALIVE = "30m"

ENABLE_TTS = True

TTS_LANGUAGE = "a"
TTS_VOICE = "am_michael"
TTS_SPEED = 1.05

HISTORY_FILE = "memory/conversation.json"

# ULTRON microphone settings
MIC_DEVICE = 2
MIC_SAMPLE_RATE = 44100
MIC_CHANNELS = 1

# Speech recognition
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"