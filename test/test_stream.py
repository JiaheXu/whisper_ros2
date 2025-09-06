import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np

class AudioProcessor(Node):
    def __init__(self):
        super().__init__('audio_processor')
        self.rate = 16000  # Sample rate
        self.channels = 1  # Mono
        self.get_logger().info("Receiving and playing audio stream...")
        
        self.input_stream = sd.RawInputStream(samplerate=self.rate, channels=self.channels, dtype = 'int16', callback=self.callback)
        self.output_stream = sd.RawOutputStream(samplerate=self.rate, channels=self.channels, dtype = 'int16')
        
        self.output_stream.start()
        self.input_stream.start()

    def callback(self, indata, frames, time, status):
        if status:
            self.get_logger().warn(f"Stream status: {status}")
        array = np.frombuffer(indata, dtype=np.int16)
        print("array: ", np.max(array), " ", np.min(array))
        self.output_stream.write(indata)


def main(args=None):
    rclpy.init(args=args)
    audio_processor = AudioProcessor()
    rclpy.spin(audio_processor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

