import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets"
DATA_ROOT = str(Path(os.environ.get("ADACLIP_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve())
