"""Download the competition dataset via kagglehub, using KAGGLE_API_TOKEN from .env."""

import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import kagglehub

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"


def main() -> None:
    path = kagglehub.competition_download("filament-segmentation-2026")
    print("Downloaded to cache:", path)

    # kagglehub's cache dir already contains a MAGFiLO_1.0_Kaggle_2026/ wrapper,
    # so copy *that* directory itself, not its parent, to avoid double nesting.
    src = Path(path) / "MAGFiLO_1.0_Kaggle_2026"
    if DEST.exists():
        print("Destination already exists, skipping copy:", DEST)
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, DEST)
    print("Copied to:", DEST)


if __name__ == "__main__":
    main()
