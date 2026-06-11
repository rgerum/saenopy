import os
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path.home() / "Library" / "Caches" / "Saenopy" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
