"""Download piano MIDI datasets into data/raw/.

Primary path is Kaggle (needs an API token). A credentials-free fallback is
also included, because MAESTRO's *MIDI-only* archive is a plain 58 MB HTTP
download -- no account required.

Usage
-----
    python data/fetch_kaggle.py --list
    python data/fetch_kaggle.py --dataset piano-midi-de
    python data/fetch_kaggle.py --dataset maestro-v3           # needs a token
    python data/fetch_kaggle.py --direct maestro-v3-midi        # no token
    python data/fetch_kaggle.py --all --small-first

Kaggle token setup (one time):
    * Kaggle -> Account -> Create New API Token -> saves kaggle.json
    * move it to ~/.kaggle/kaggle.json   (chmod 600 on Linux/macOS)
    * or just export the two values:
        KAGGLE_USERNAME=yourname  KAGGLE_KEY=xxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

KAGGLE_DATASETS = {
    "piano-midi-de": {
        "slug": "soumikrakshit/classical-music-midi",
        "about": "Classical piano only, scraped from piano-midi.de; 19 composers.",
        "size": "~5 MB",
        "license": "Free for non-commercial use (piano-midi.de terms)",
        "why": "Best first dataset: small, 100% solo piano, human performances.",
    },
    "maestro-midi": {
        "slug": "kritanjalijain/maestropianomidi",
        "about": "MAESTRO piano MIDI mirrors (part of the official corpus).",
        "size": "~60 MB+",
        "license": "CC BY-NC-SA 4.0 (non-commercial, share-alike)",
        "why": "Virtuosic Disklavier performances; includes sustain pedal.",
    },
    "maestro-v2": {
        "slug": "jackvial/themaestrodatasetv2",
        "about": "MAESTRO v2, MIDI + audio split across several archives.",
        "size": "hundreds of MB",
        "license": "CC BY-NC-SA 4.0",
        "why": "Same corpus as above; pick whichever mirror is intact.",
    },
    "giantmidi-piano": {
        "slug": "pictureinthenoise/music-generation-with-giantmidi-piano",
        "about": "GiantMIDI-Piano: ~10.8k piano MIDI files, ~1.2k hours.",
        "size": "~1 GB",
        "license": "See dataset page; derived from IMSLP recordings",
        "why": "Big variety after you have a pipeline that works.",
    },
    "midi-classic-music": {
        "slug": "blanderbuss/midi-classic-music",
        "about": "3.9k classical MIDI files, 175 composers, NOT piano-only.",
        "size": "~20 MB",
        "license": "See dataset page",
        "why": "Useful later; needs the instrument filtering step.",
    },
    "classical-piano-midi": {
        "slug": "gautamgc75/classical-piano-midi-music",
        "about": "~6k classical piano MIDI files.",
        "size": "~190 MB",
        "license": "See dataset page",
        "why": "Middle ground between the two above.",
    },
}

DIRECT_SOURCES = {
    "maestro-v3-midi": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "sha256": "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c",
        "size": "~58 MB",
        "license": "CC BY-NC-SA 4.0",
    },
    "maestro-v2-midi": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
        "sha256": "ec2cc9d94886c6b376db1eaa2b8ad1ce62ff9f0a28b3744782b13163295dadf3",
        "size": "~59 MB",
        "license": "CC BY-NC-SA 4.0",
    },
    "maestro-v1-midi": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip",
        "sha256": "f620f9e1eceaab8beea10617599add2e9c83234199b550382a2f603098ae7135",
        "size": "~47 MB",
        "license": "CC BY-NC-SA 4.0",
    },
}


def print_registry() -> None:
    print("Kaggle datasets (python data/fetch_kaggle.py --dataset <key>)\n")
    for key, meta in KAGGLE_DATASETS.items():
        print(f"  {key:<20} {meta['slug']}")
        print(f"  {'':<20} {meta['about']}")
        print(f"  {'':<20} size {meta['size']} | licence: {meta['license']}")
        print(f"  {'':<20} why: {meta['why']}\n")
    print("Direct downloads, no Kaggle account needed (--direct <key>)\n")
    for key, meta in DIRECT_SOURCES.items():
        print(f"  {key:<20} {meta['size']} | {meta['license']}")


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home = Path.home() / ".kaggle"
    return (home / "kaggle.json").exists() or (home / "access_token").exists()


def credential_help() -> str:
    return (
        "No Kaggle credentials found.\n"
        "  * Kaggle -> Account -> Create New API Token, then save the file as\n"
        f"      {Path.home() / '.kaggle' / 'kaggle.json'}\n"
        "  * or set KAGGLE_USERNAME and KAGGLE_KEY in the environment.\n"
        "  * The Kaggle dataset pages can also be downloaded by hand from a browser;\n"
        f"    drop the extracted files into {RAW_DIR} and skip this script.\n"
        "  * Still blocked? Use the token-free mirror instead:\n"
        "      python data/fetch_kaggle.py --direct maestro-v3-midi"
    )


def fetch_kaggle(key: str) -> Path:
    meta = KAGGLE_DATASETS[key]
    slug = meta["slug"]
    dest = RAW_DIR / key
    if dest.exists() and any(dest.rglob("*")):
        print(f"Already present: {dest}")
        return dest

    if not has_kaggle_credentials():
        raise SystemExit(credential_help())

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        print(f"Downloading {slug} via kagglehub ...")
        cached = Path(kagglehub.dataset_download(slug))
        shutil.copytree(cached, dest, dirs_exist_ok=True)
        print(f"Saved to {dest}")
        return dest
    except ImportError:
        pass

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Neither kagglehub nor the kaggle CLI is installed. Pick one:\n"
            "  pip install kagglehub\n"
            "  pip install kaggle\n" + credential_help()
        )

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"kaggle CLI exited with {result.returncode}. Check that you have "
            f"accepted the dataset's rules on its Kaggle page."
        )
    return dest


def fetch_direct(key: str) -> Path:
    meta = DIRECT_SOURCES[key]
    dest = RAW_DIR / key
    if dest.exists() and any(dest.rglob("*.mid*")):
        print(f"Already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / f"{key}.zip"
    print(f"Downloading {meta['url']} ({meta['size']}) ...")

    digest = hashlib.sha256()

    def report(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 200 == 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    with urllib.request.urlopen(meta["url"]) as response, open(archive, "wb") as fh:
        while chunk := response.read(1 << 16):
            fh.write(chunk)
            digest.update(chunk)
            report(fh.tell() // (1 << 16), 1 << 16, int(response.headers.get("Content-Length") or 0))
    print()

    actual = digest.hexdigest()
    if actual != meta["sha256"]:
        archive.unlink(missing_ok=True)
        raise SystemExit(
            f"Checksum mismatch -- download discarded.\n  expected {meta['sha256']}\n  got      {actual}"
        )
    print("Checksum OK")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()
    print(f"Extracted to {dest}")
    return dest


def summarise(root: Path) -> None:
    midi = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
    csvs = sorted(p for p in root.rglob("*.csv"))
    total = sum(p.stat().st_size for p in midi) / 1e6
    print(f"\n{root}: {len(midi)} MIDI files, {total:.1f} MB on disk")
    for csv in csvs:
        print(f"  metadata: {csv.relative_to(root)}")
        try:
            head = csv.read_text(errors="ignore").splitlines()[:2]
            print(f"    columns: {head[0]}")
            if len(head) > 1:
                print(f"    example: {head[1]}")
        except OSError as exc:
            print(f"    (could not read: {exc})")
    if midi:
        print(f"  first file: {midi[0].relative_to(root)}")
    print("\nNext: python data/prepare_midi.py --input data/raw --output data/processed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show available datasets and licences")
    parser.add_argument("--dataset", action="append", default=[], metavar="KEY",
                        help="Kaggle dataset key to download (repeatable)")
    parser.add_argument("--direct", action="append", default=[], metavar="KEY",
                        help="token-free direct source key (repeatable)")
    parser.add_argument("--all", action="store_true", help="every Kaggle dataset in the registry")
    parser.add_argument("--small-first", action="store_true",
                        help="with --all, download smallest datasets first")
    args = parser.parse_args(argv)

    if args.list or not (args.dataset or args.direct or args.all):
        print_registry()
        return 0

    datasets = list(args.dataset)
    if args.all:
        datasets = list(KAGGLE_DATASETS)
    if args.small_first:
        datasets.sort(key=lambda k: KAGGLE_DATASETS[k]["size"])

    for key in datasets:
        if key not in KAGGLE_DATASETS:
            raise SystemExit(f"Unknown dataset {key!r}. Known: {', '.join(KAGGLE_DATASETS)}")
        fetch_kaggle(key)

    for key in args.direct:
        if key not in DIRECT_SOURCES:
            raise SystemExit(f"Unknown direct source {key!r}. Known: {', '.join(DIRECT_SOURCES)}")
        fetch_direct(key)

    summarise(RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
