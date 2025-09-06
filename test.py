import whisper

# Load model
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Transcribe audio
for i in range(1,5):
    result = model.transcribe(f"{i}.wav")
    print(f"{i}th data:")
    print(result["text"])

