"""CPU forward benchmark on zero tensors; NOT a real-action accuracy test."""
from pathlib import Path
import sys
import threading
import time
import statistics
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import torch
from src.motionmirror.model import configured_model

process = psutil.Process()
peak = [process.memory_info().rss]
stop = threading.Event()

def monitor():
    while not stop.wait(.01):
        peak[0] = max(peak[0], process.memory_info().rss)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    start = time.perf_counter()
    model = configured_model()
    load_time = time.perf_counter() - start
    clip = torch.zeros(1, 3, model.spec.num_frames, model.spec.frame_size, model.spec.frame_size)
    with torch.inference_mode():
        model.network(clip)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            model.network(clip)
            times.append(time.perf_counter() - start)
    print(json.dumps({"input": "zero tensor, not a real action video", "cpu": __import__('platform').processor(),
        "threads": torch.get_num_threads(), "load_seconds": load_time,
        "forward_seconds": times, "median_forward_seconds": statistics.median(times),
        "parameter_mb": model.parameter_mb, "sampled_peak_process_rss_mb": peak[0] / 1024**2}, indent=2))
finally:
    stop.set()
    thread.join()
