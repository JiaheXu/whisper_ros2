import sounddevice as sd
from scipy.io.wavfile import read

# Path to your WAV file
filename = "example.wav"

# Read the WAV file
samplerate, data = read(filename)

# Play the audio
print("Playing audio...")
sd.play(data, samplerate)

# Wait until playback is finished
sd.wait()
print("Playback finished.")