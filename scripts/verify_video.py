"""Run the supplied model on a REAL action video and report CPU/RAM measurements."""
import argparse
import json
from pathlib import Path
import sys
import time
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from src.motionmirror.model import configured_model
from src.motionmirror.inference import predict

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("--mode", choices=["Standard", "Enhanced"], default="Standard")
args = parser.parse_args()
process = psutil.Process()
peak = [process.memory_info().rss]
stop = threading.Event()

def monitor():
    while not stop.wait(.02):
        peak[0] = max(peak[0], process.memory_info().rss)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    start = time.perf_counter()
    model = configured_model()
    load_seconds = time.perf_counter() - start
    result = predict(args.video, model, args.mode)
    print(json.dumps({"video": args.video.name, "action": result.action,
        "scores": dict(zip(result.labels, result.scores)), "clips": result.clips,
        "load_seconds": load_seconds, "processing_seconds": result.elapsed_seconds,
        "parameter_mb": model.parameter_mb, "sampled_peak_process_rss_mb": peak[0] / 1024**2}, indent=2))
finally:
    stop.set()
    thread.join()
