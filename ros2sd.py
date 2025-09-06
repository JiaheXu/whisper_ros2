import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np
from audio_common_msgs.msg import Audio, AudioData,AudioStamped

class AudioSubscriber(Node):
    def __init__(self):
        super().__init__('audio_subscriber')
        self.subscription = self.create_subscription(AudioStamped, 'audio', self.callback, 10)
        self.rate = 16000  # Sample rate
        self.channels = 1  # Stereo
        self.output_stream = sd.RawOutputStream(samplerate=self.rate, channels=self.channels, dtype = 'int16')
        self.output_stream.start()
        self.get_logger().info("Listening to audio stream...")


    def callback(self, msg):
        #self.get_logger().info("in callback")
        audio_data = np.array(msg.audio.audio_data.int16_data, dtype=np.int16)
        audio_data = audio_data.reshape(-1, self.channels)  # Ensure correct shape
        self.output_stream.write(audio_data)


def main(args=None):
    rclpy.init(args=args)
    audio_subscriber = AudioSubscriber()
    rclpy.spin(audio_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

