"""Report independent results and errors from four sequential checkpoint attempts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.motionmirror.comparison import compare_all_models, ComparisonError
from src.motionmirror.preprocessing import VideoError

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path, help="A real supported action video")
parser.add_argument("--output", type=Path, help="Save the full result JSON and print a compact summary")
args = parser.parse_args()
try:
    result = compare_all_models(args.video)
except (ComparisonError, VideoError) as exc:
    parser.exit(1, f"{exc}\n")
payload = {"video": args.video.name, "device": "cpu", **asdict(result)}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(result.models)}/4 predictions; {len(result.errors)} errors; "
          f"{result.total_seconds:.3f}s total; {result.peak_rss_mb:.1f} MiB sampled peak RSS")
    for item in result.models:
        print(f"{item.architecture}: {item.prediction.forward_seconds:.3f}s inference; "
              f"{item.load_seconds:.3f}s loading; {item.prediction.elapsed_seconds:.3f}s processing")
    print(f"Full results: {args.output}")
else:
    print(json.dumps(payload, indent=2))
if result.errors:
    sys.exit(1)
