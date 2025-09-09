import torch
import os
import argparse
import time  # <-- import time
from whisper_trt import load_trt_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["tiny.en", "base.en", "small.en"])
    parser.add_argument("audio", type=str)
    parser.add_argument("--backend", type=str, choices=["whisper", "whisper_trt", "faster_whisper"], default="whisper_trt")
    args = parser.parse_args()

    start_time = time.time()  # <-- start timer

    if args.backend == "whisper":

        from whisper import load_model

        model = load_model(args.model)
        result = model.transcribe(args.audio)
        
    elif args.backend == "whisper_trt":

        from whisper_trt import load_trt_model

        model = load_trt_model(args.model)
        result = model.transcribe(args.audio)

    elif args.backend == "faster_whisper":

        from faster_whisper import WhisperModel

        model = WhisperModel(args.model)
        segs, info = model.transcribe(args.audio)
        text = "".join(seg.text for seg in segs)
        result = {"text": text}

    else:
        raise RuntimeError("Unsupported backend.")

    end_time = time.time()  # <-- end timer
    elapsed = end_time - start_time

    print(f"Result: {result['text']}")
    print(f"Transcription running time: {elapsed:.2f} seconds")

# python3 transcribe.py tiny.en example.wav --backend whisper_trt