import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import time  # <-- import time
from whisper_trt import load_trt_model
from whisper import load_model

class ASRNode(Node):
    def __init__(self):
        super().__init__("ASRNode")

        self.use_channel = 0
        # self.asr_model = load_model("base.en")
        self.asr_model = load_trt_model("base.en")
        self.asr_model.transcribe(np.zeros(1536, dtype=np.float32))  # warmup

        self.speech_sub = self.create_subscription(
            String,
            "/speech_audio",
            self.speech_callback,
            10
        )

        audio_file = '/home/developer/javis_ws/whisper_ws/src/whisper_ros2/example.wav'
        start_time = time.time()  # <-- start timer
        result = self.asr_model.transcribe(audio_file)

        end_time = time.time()  # <-- end timer
        elapsed = end_time - start_time

        print(f"Result: {result['text']}")
        print(f"Transcription running time: {elapsed:.2f} seconds")

        self.text_pub = self.create_publisher(String, "/speech", 10)

        self.get_logger().info("ASRNode started")

    def speech_callback(self, msg: String):
        # reconstruct audio segment from hex strings
        chunk_bytes_list = [bytes.fromhex(x) for x in eval(msg.data)]
        audio_numpy = np.concatenate([
            np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
            for b in chunk_bytes_list
        ])
        text = self.asr_model.transcribe(audio_numpy)["text"]
        self.get_logger().info(f"ASR result: {text}")
        self.text_pub.publish(String(data=text))

def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
