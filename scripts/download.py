import os
from pathlib import Path

os.environ.setdefault("KAGGLEHUB_CACHE", str(Path(__file__).resolve().parents[1] / "data/kagglehub"))

import kagglehub

path = kagglehub.competition_download("multiple-object-video-detection-hw-2-ucu-ai-2026")
print("Path to competition files:", path)
