# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
# from whisper_trt_pipeline import WhisperTRTPipeline
from audio_common_msgs.msg import Audio, AudioStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import io
import wave
import numpy as np
from whisper_trt.vad import load_vad
from whisper_trt import load_trt_model, set_cache_dir
from whisper import load_model
import time
import pyaudio
import socketio
# import eventlet
import asyncio
import uvicorn
import starlette
# from aiohttp import web
import socketio
import threading
# from multiprocessing import Process, Queue, Event
from threading import Thread, Event
from queue import Queue
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

def audio_numpy_from_bytes(audio_bytes: bytes):
    audio = np.fromstring(audio_bytes, dtype=np.int16)
    return audio


def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, 
                      num_channels: int = 6):
    return audio_numpy[channel_index::num_channels]


def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768


@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None


@dataclass
class AudioSegment:
    chunks: AudioChunk

class WhisperTRTNode(Node):
    def __init__(self):
        super().__init__('WhisperTRTNode')

        self.declare_parameter("model", "small.en")
        self.declare_parameter("backend", "whisper_trt") 

        # TODO: remove placeholder default
        self.declare_parameter("cache_dir", "data")#rclpy.Parameter.Type.STRING)
        self.declare_parameter("vad_window", 3)

        #self.declare_parameter("mic_device_index", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("mic_device_index", 0)
        self.declare_parameter("mic_sample_rate", 16000)
        self.declare_parameter("mic_channels", 6)
        self.declare_parameter("mic_bitwidth", 2)
        self.declare_parameter("mic_channel_for_asr", 0)

        self.declare_parameter("speech_topic", "/speech")

        qos_profile = QoSProfile(
            depth=10,  # Depth of the message queue
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Reliability setting
            durability=QoSDurabilityPolicy.VOLATILE  # Durability setting
        )

        self.logger = self.get_logger()
        self.sample_rate = 16000
        self.max_filter_window = 8
        self.use_channel = 0
        self.speech_threshold = 0.7
        self.speech_chunks = []
        self.chunk_size = 2048

        ###############################################################
        # ROS2 audio msg -> audio chunk (audio_queue) -> VAD model -> speech chunk (speech_queue) -> text 
        ###############################################################

        self.audio_queue = Queue()
        self.speech_queue = Queue()

        ###############################################################
        # VAD model, tell if a chunk of audio is speech or not
        ###############################################################
        self.vad_model = load_vad("/home/developer/javis_ws/whisper_ws/src/whisper_ros2/model/silero_vad.onnx")
        print("finished vad model loading")
        # warmup run
        self.vad_model(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)
        self.prev_is_voice = False
        self.max_filter_window = deque(maxlen=self.max_filter_window)
        
        ###############################################################
        # ASR model, convert speech to text
        ###############################################################
        #self.asr_model = load_trt_model("tiny.en")
        self.asr_model = load_model("small.en")
        # warmup
        self.asr_model.transcribe(np.zeros(1536, dtype=np.float32))
        print("finished asr_model loading")

        # Start a separate thread for processing messages
        self.vad_thread = threading.Thread(target=self.run_vad)
        self.vad_thread.daemon = True  # Ensures thread exits when node stops
        self.vad_thread.start()

        # Start a separate thread for processing messages
        self.asr_thread = threading.Thread(target=self.run_asr)
        self.asr_thread.daemon = True  # Ensures thread exits when node stops
        self.asr_thread.start()
        self.current_chunk = []

        self.speech_publisher = self.create_publisher(
            String, 
            "speech",
            10
        )

        self.audio_sub = self.create_subscription(
            AudioStamped,
            "/audio",
            self.audio_callback,
            qos_profile
        )


    def run_vad(self):


        while rclpy.ok():
            if not self.audio_queue.empty():
                # msg_data = self.audio_queue.get()  # Retrieve message from queue
                # self.get_logger().info(f"processing !!!")
                audio_chunk = self.audio_queue.get()
                # print("audio_chunk: ", audio_chunk.audio_numpy_normalized[self.use_channel].shape )
                # voice_prob = 0.0
                voice_prob = float( self.vad_model(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0] )
                # print("voice_prob: ", voice_prob)
                chunk = AudioChunk(
                    audio_raw=audio_chunk.audio_raw,
                    audio_numpy=audio_chunk.audio_numpy,
                    audio_numpy_normalized=audio_chunk.audio_numpy_normalized,
                    voice_prob=voice_prob
                )

                self.max_filter_window.append(chunk)

                is_voice = any(c.voice_prob > self.speech_threshold for c in self.max_filter_window)                    
                if is_voice > self.prev_is_voice:
                    self.speech_chunks = [chunk for chunk in self.max_filter_window]
                    # start voice
                    self.speech_chunks.append(chunk)

                elif is_voice < self.prev_is_voice:
                    # end voice
                    segment = AudioSegment(chunks=self.speech_chunks)
                    self.speech_queue.put(segment)

                elif is_voice:
                    # continue voice
                    self.speech_chunks.append(chunk)

                self.prev_is_voice = is_voice

                # print("prev_is_voice: ", is_voice)
            else:
                time.sleep(0.05) 
                # time.sleep(0.1)  # Prevent CPU overuse when queue is empty

    def run_asr(self):
        while rclpy.ok():
            if not self.speech_queue.empty():        
                print("in asr callback")
                speech_segment = self.speech_queue.get()
                audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])
                text = self.asr_model.transcribe(audio)['text']
                # self.get_logger().info(text)
                print("text: ", text)
            else:
                time.sleep(0.01) 
        # return text

    def get_audio(self, audio_raw, num_channels=2):
        # self.pipeline.audio_chunks
        # audio_raw = stream.read(self.chunk_size)
        audio_numpy = audio_numpy_from_bytes(audio_raw)
        audio_numpy = np.stack([audio_numpy_slice_channel(audio_numpy, i, num_channels) for i in range(num_channels)])
        # print("audio_raw: ", audio_raw)
        audio_numpy_normalized = audio_numpy_normalize(audio_numpy)
        # print("audio_numpy_normalized: ", np.max(audio_numpy_normalized)," ", np.min(audio_numpy_normalized) )
        audio = AudioChunk(
            audio_raw=audio_raw,
            audio_numpy=audio_numpy,
            audio_numpy_normalized=audio_numpy_normalized
        )

        self.audio_queue.put(audio)
        if( self.audio_queue.qsize()%20 == 0 ):
            print("queueu size: ", self.audio_queue.qsize())

    def audio_callback(self, msg):
        # self.get_logger().info("Received audio data.")
        # audio_raw = io.BytesIO(msg.audio.audio_data.int16_data)
        # print("data: ", len(msg.audio.audio_data.int16_data) )
        if(self.current_chunk is None):
            self.current_chunk = np.array(msg.audio.audio_data.int16_data, np.int16)
        else:
            self.current_chunk = np.append( self.current_chunk, np.array(msg.audio.audio_data.int16_data, np.int16) )
        # print("chunk: ", self.current_chunk.shape[0])

        if( self.current_chunk.shape[0] > self.chunk_size):

            audio_raw = self.current_chunk.tobytes()
            # print("audio_raw: ", len(audio_raw) )
            num_channels = msg.audio.info.channels
            format = msg.audio.info.format
            rate = msg.audio.info.rate
            chunk = msg.audio.info.chunk
            # print("chunk: ", chunk)
            self.get_audio(audio_raw, num_channels)
            self.current_chunk = None

def main(args=None):
    rclpy.init(args=args)
    node = WhisperTRTNode()
    print("finished init")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
