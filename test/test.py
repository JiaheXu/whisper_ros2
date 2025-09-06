import sounddevice as sd

# Set parameters
SAMPLE_RATE = 16000  # Sample rate (Hz)
CHANNELS = 1         # Number of channels (1 for mono, 2 for stereo)
BLOCKSIZE = 1024     # Block size for buffer

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Status: {status}")  # Print any errors or warnings
    outdata[:] = indata  # Send captured audio to the output

try:
    # Open a stream for simultaneous input and output
    with sd.Stream(channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                   callback=audio_callback):
        print("Press Ctrl+C to stop...")
        sd.sleep(100 * 1000)  # Keep the stream open for 10 seconds
except KeyboardInterrupt:
    print("Audio stream stopped.")
except Exception as e:
    print(f"An error occurred: {e}")
