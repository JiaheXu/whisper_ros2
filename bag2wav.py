
import rclpy
import numpy as np
import os
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from scipy.io.wavfile import write as wav_write

BAG_PATH = '/path/to/your/ros2_bag'  # Replace with your actual bag path
TOPIC_NAME = '/audio'
OUTPUT_WAV = 'output.wav'

def read_audio_topic_to_wav():
    reader = SequentialReader()
    storage_options = StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    msg_type_str = type_map[TOPIC_NAME]
    msg_type = get_message(msg_type_str)

    audio_data = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == TOPIC_NAME:
            msg = deserialize_message(data, msg_type)
            audio_data.extend(msg.audio.audio_data.int16_data)

    if not audio_data:
        print(f"No audio data found in topic {TOPIC_NAME}")
        return

    audio_array = np.array(audio_data, dtype=np.int16)
    sample_rate = msg.audio.info.rate  # e.g. 16000

    wav_write(OUTPUT_WAV, sample_rate, audio_array)
    print(f"Saved WAV file to {OUTPUT_WAV}")

if __name__ == '__main__':
    rclpy.init()
    read_audio_topic_to_wav()
    rclpy.shutdown()
