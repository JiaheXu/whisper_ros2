import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np
from audio_common_msgs.msg import Audio, AudioData, AudioInfo, AudioStamped

class AudioPublisher(Node):
    def __init__(self):
        super().__init__('audio_publisher')
        self.publisher_ = self.create_publisher(AudioStamped, 'audio', 10)
        self.rate = 16000  # Sample rate
        self.channels = 1  # Mono
        self.chunk_size = 4096  # Buffer size
        self.audio_info = AudioInfo()
        self.audio_info.format = 8
        self.audio_info.channels = self.channels
        self.audio_info.rate = self.rate        
        self.audio_info.chunk = self.chunk_size
        
        self.get_logger().info("Starting audio stream...")
        with sd.RawInputStream(samplerate=self.rate, channels=self.channels, dtype = 'int16', callback=self.callback):
        # with sd.RawInputStream(samplerate=self.rate, channels=self.channels, callback=self.callback):
            rclpy.spin(self)

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")  # Print any errors or warnings
        # self.output_stream.write(indata)
        array = np.frombuffer(indata, dtype=np.int16)
        audio_msg = AudioStamped()
        audio_msg.header.stamp = self.get_clock().now().to_msg()
        audio_msg.audio.audio_data.int16_data = array.tolist()
        audio_msg.audio.info = self.audio_info
        
        self.publisher_.publish(audio_msg)

def main(args=None):
    rclpy.init(args=args)
    audio_publisher = AudioPublisher()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
