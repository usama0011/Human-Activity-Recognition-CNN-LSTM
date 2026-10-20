"""Download/extract UCI HAR data; nothing is uploaded to GitHub."""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

# The public UCI dataset archive is sometimes served as an outer ZIP containing another ZIP.
UCI_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"


def extract_safe(archive, destination):
    destination = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive path: {info.filename}")
    archive.extractall(destination)


def unpack(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as outer:
        inner_names = [name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip")]
        if inner_names:
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_names[0]))) as inner:
                extract_safe(inner, destination)
        else:
            extract_safe(outer, destination)
    if not (destination / "UCI HAR Dataset" / "train" / "Inertial Signals").is_dir():
        raise RuntimeError("Extracted ZIP does not contain expected raw inertial signals")
    print(f"Dataset ready: {destination / 'UCI HAR Dataset'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare official UCI HAR dataset")
    parser.add_argument("--zip", type=Path, dest="archive", help="Previously downloaded official UCI ZIP")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.archive:
        if not args.archive.is_file():
            parser.error(f"File not found: {args.archive}")
        unpack(args.archive, args.output)
    else:
        print("Downloading official UCI HAR archive (~58 MB)...")
        try:
            with urllib.request.urlopen(UCI_URL, timeout=90) as response:
                archive = io.BytesIO(response.read())
            unpack(archive, args.output)
        except Exception as error:
            raise SystemExit(
                f"Automatic download failed: {error}\n"
                "Download 'UCI HAR Dataset.zip' from the official UCI website, then run:\n"
                "python prepare_data.py --zip 'path/to/UCI HAR Dataset.zip'"
            ) from error


if __name__ == "__main__":
    main()
