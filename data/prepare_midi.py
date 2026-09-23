"""Turn a folder of messy MIDI into a clean, piano-only dataset for next-note
prediction.

What it does per file:
  * keeps piano channels only (GM programs 1-8), always dropping channel 10/drums
  * optionally prolongs notes according to the sustain pedal (CC64)
  * collapses spurious repeated note onsets (same pitch at the same instant)
  * derives duration by integrating tempo, so ritenuto endings don't distort it
  * drops out-of-range notes (default: the 88 piano keys, MIDI 21-108)
  * quantises note onsets to a grid so the model sees clean time steps
  * drops near-duplicate pieces and too-short/too-empty files
  * skips previously *generated* files so the model never trains on its own output
  * writes fixed-resolution MIDI into train/val/test folders plus a manifest

Splits follow MAESTRO's official metadata CSV when one is found in the input
tree, otherwise a deterministic hash split is used. Either way the split is
stable across runs, and `make_sequences` later keeps windows inside one split.

Usage
-----
    python data/prepare_midi.py --self-test
    python data/prepare_midi.py --input data/raw --output data/processed
    python data/prepare_midi.py --input data/raw --limit 50 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

try:
    import mido
except ImportError:
    raise SystemExit("mido is required:  pip install mido")

PIANO_PROGRAMS = set(range(8))
DRUM_CHANNEL = 9
MIDI_SUFFIXES = {".mid", ".midi"}
DEFAULT_EXCLUDES = ("Melody_Generated*", "output.mid", "generated*.mid", "sample*.mid")


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    velocity: int
    channel: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    grid: float = 0.25
    min_notes: int = 30
    min_duration: float = 10.0
    pitch_low: int = 21
    pitch_high: int = 108
    apply_sustain: bool = True
    allow_default_program: bool = True
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42
    limit: int | None = None
    dry_run: bool = False
    plot: bool = True
    exclude: tuple[str, ...] = ()


def extract_notes(midi: mido.MidiFile, apply_sustain: bool) -> tuple[list[Note], float, float]:
    """Flatten a MIDI file into absolute-time notes, merging all tracks.

    Returns (notes, initial tempo in bpm, elapsed seconds). Note times stay in
    beats so quantisation is tempo-independent, while elapsed time is *integrated*
    across every tempo change. Taking the first or last tempo instead is wildly
    wrong for performances that ritenuto at the end -- classical MIDI frequently
    ends at 14 bpm, which would report a 5 minute piece as 56 minutes.

    Approximation: a note released while the pedal is down is extended to the
    pedal release time. Rapid retriggers under the pedal are collapsed, which is
    musically harmless for a note-event model.
    """
    programs: dict[int, int] = {}
    pedal_down: dict[int, bool] = {}
    open_notes: dict[tuple[int, int], tuple[float, int]] = {}
    deferred: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    notes: list[Note] = []

    beats = 0.0
    seconds = 0.0
    bpm = 120.0
    initial_bpm: float | None = None
    per_beat = midi.ticks_per_beat or 480

    for msg in mido.merge_tracks(midi.tracks):
        delta_beats = msg.time / per_beat
        beats += delta_beats
        seconds += delta_beats * 60.0 / bpm
        kind = msg.type
        if kind == "set_tempo":
            bpm = 60_000_000 / msg.tempo
            if initial_bpm is None:
                initial_bpm = bpm
        elif kind == "program_change":
            programs[msg.channel] = msg.program
        elif kind == "control_change" and msg.control == 64:
            down = msg.value >= 64
            if not down and pedal_down.get(msg.channel):
                for pitch, start, velocity in deferred.pop(msg.channel, []):
                    notes.append(Note(pitch, start, beats, velocity, msg.channel))
            pedal_down[msg.channel] = down
        elif kind == "note_on" and msg.velocity > 0:
            key = (msg.channel, msg.note)
            if key in open_notes:
                start, velocity = open_notes.pop(key)
                notes.append(Note(msg.note, start, beats, velocity, msg.channel))
            open_notes[key] = (beats, msg.velocity)
        elif kind == "note_off" or (kind == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key not in open_notes:
                continue
            start, velocity = open_notes.pop(key)
            if apply_sustain and pedal_down.get(msg.channel):
                deferred[msg.channel].append((msg.note, start, velocity))
            else:
                notes.append(Note(msg.note, start, beats, velocity, msg.channel))

    for (channel, pitch), (start, velocity) in open_notes.items():
        notes.append(Note(pitch, start, beats, velocity, channel))
    for channel, pending in deferred.items():
        for pitch, start, velocity in pending:
            notes.append(Note(pitch, start, beats, velocity, channel))

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, initial_bpm or 120.0, seconds


def is_piano_channel(channel: int, programs: dict[int, int], allow_default: bool) -> bool:
    if channel == DRUM_CHANNEL:
        return False
    if channel in programs:
        return programs[channel] in PIANO_PROGRAMS
    return allow_default


def keep_piano_channels(midi: mido.MidiFile, notes: list[Note], allow_default: bool) -> tuple[list[Note], int]:
    programs: dict[int, int] = {}
    for msg in mido.merge_tracks(midi.tracks):
        if msg.type == "program_change":
            programs[msg.channel] = msg.program
    kept = [n for n in notes if is_piano_channel(n.channel, programs, allow_default)]
    return kept, len(notes) - len(kept)


def quantize(notes: list[Note], grid: float) -> list[Note]:
    if grid <= 0:
        return notes
    quantised = []
    for n in notes:
        start = round(n.start / grid) * grid
        end = round(n.end / grid) * grid
        if end <= start:
            end = start + grid
        quantised.append(replace(n, start=round(start, 6), end=round(end, 6)))
    return quantised


def collapse_duplicate_onsets(notes: list[Note]) -> tuple[list[Note], int]:
    """Remove repeated (pitch, onset) pairs.

    A piano key cannot strike twice at the same instant, so these are editing
    artifacts (a duplicate note_on with no note_off in between). 17 of the 295
    pieces in `music/` had them, up to 15% of their notes.
    """
    best: dict[tuple[int, float], Note] = {}
    for n in notes:
        key = (n.pitch, round(n.start, 3))
        current = best.get(key)
        if current is None or n.duration > current.duration:
            best[key] = n
    collapsed = sorted(best.values(), key=lambda n: (n.start, n.pitch))
    return collapsed, len(notes) - len(collapsed)


def write_clean_midi(notes: list[Note], path: Path, bpm: float, ticks_per_beat: int = 480) -> None:
    """Write a single-track, single-program piano MIDI at a fixed resolution."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))

    events = []
    for n in notes:
        start_tick = max(0, int(round(n.start * ticks_per_beat)))
        end_tick = max(start_tick + 1, int(round(n.end * ticks_per_beat)))
        events.append((start_tick, 1, n.pitch, n.velocity))
        events.append((end_tick, 0, n.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))

    previous_tick = 0
    for tick, is_on, pitch, velocity in events:
        delta = tick - previous_tick
        previous_tick = tick
        if is_on:
            track.append(mido.Message("note_on", note=pitch, velocity=max(1, min(127, velocity)), time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def load_maestro_splits(input_dir: Path) -> dict[str, str]:
    """Read MAESTRO's official split column so we never straddle the boundary."""
    for csv_path in input_dir.rglob("*.csv"):
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
                reader = csv.DictReader(fh)
                fields = {f.strip().lower() for f in (reader.fieldnames or [])}
                if not {"split", "midi_filename"} <= fields:
                    continue
                lower = {f.strip().lower(): f for f in reader.fieldnames or []}
                mapping = {}
                for row in reader:
                    name = Path(str(row[lower["midi_filename"]])).name.lower()
                    mapping[name] = str(row[lower["split"]]).strip().lower()
                if mapping:
                    print(f"Using official splits from {csv_path} ({len(mapping)} entries)")
                    return mapping
        except OSError:
            continue
    return {}


def assign_split(name: str, official: dict[str, str], cfg: Config) -> str:
    key = Path(name).name.lower()
    if key in official:
        split = official[key]
        return split if split in {"train", "validation", "test"} else "train"
    digest = hashlib.sha1(key.encode()).digest()
    roll = int.from_bytes(digest[:4], "big") % 1000 / 1000
    if roll < cfg.train_frac:
        return "train"
    if roll < cfg.train_frac + cfg.val_frac:
        return "validation"
    return "test"


def prepare(cfg: Config) -> dict:
    candidates = sorted(p for p in cfg.input_dir.rglob("*") if p.suffix.lower() in MIDI_SUFFIXES)
    files = [p for p in candidates if not any(fnmatch.fnmatch(p.name.lower(), pat.lower()) for pat in cfg.exclude)]
    kept_paths = set(files)
    excluded = [p for p in candidates if p not in kept_paths]
    if excluded:
        print(f"Excluding {len(excluded)} model-generated file(s): "
              f"{', '.join(p.name for p in excluded[:5])}")
    if cfg.limit:
        random.Random(cfg.seed).shuffle(files)
        files = sorted(files[: cfg.limit])
    if not files:
        raise SystemExit(f"No MIDI files found under {cfg.input_dir}")

    official = load_maestro_splits(cfg.input_dir)
    print(f"Found {len(files)} MIDI files under {cfg.input_dir}")

    seen_bytes: set[str] = set()
    seen_music: set[str] = set()
    rows: list[dict] = []
    rejected: list[dict] = [{"file": str(p), "reason": "generator_output"} for p in excluded]
    counts: Counter = Counter()
    total_notes = 0
    total_seconds = 0.0
    total_dups = 0
    total_nonpiano = 0
    pitch_hist: Counter = Counter()

    for path in files:
        raw = path.read_bytes()
        byte_hash = hashlib.sha1(raw).hexdigest()
        if byte_hash in seen_bytes:
            rejected.append({"file": str(path), "reason": "duplicate_bytes"})
            continue
        seen_bytes.add(byte_hash)

        try:
            midi = mido.MidiFile(str(path))
            notes, bpm, seconds = extract_notes(midi, cfg.apply_sustain)
        except Exception as exc:
            rejected.append({"file": str(path), "reason": f"parse_error: {exc}"})
            continue

        notes, dropped_nonpiano = keep_piano_channels(midi, notes, cfg.allow_default_program)
        notes = [n for n in notes if cfg.pitch_low <= n.pitch <= cfg.pitch_high]

        if len(notes) < cfg.min_notes:
            rejected.append({"file": str(path), "reason": f"only_{len(notes)}_notes"})
            continue

        notes, dup_removed = collapse_duplicate_onsets(notes)
        notes = quantize(notes, cfg.grid)
        notes, dup_from_grid = collapse_duplicate_onsets(notes)
        dup_removed += dup_from_grid
        if seconds < cfg.min_duration:
            rejected.append({"file": str(path), "reason": f"only_{seconds:.1f}s"})
            continue

        fingerprint = hashlib.sha1(
            repr([(n.pitch, round(n.start, 2)) for n in notes]).encode()
        ).hexdigest()
        if fingerprint in seen_music:
            rejected.append({"file": str(path), "reason": "duplicate_music"})
            continue
        seen_music.add(fingerprint)

        split = assign_split(path.name, official, cfg)
        out_path = cfg.output_dir / split / f"{path.stem[:80]}.mid"
        if not cfg.dry_run:
            write_clean_midi(notes, out_path, bpm)

        counts[split] += 1
        total_notes += len(notes)
        total_seconds += seconds
        total_dups += dup_removed
        total_nonpiano += dropped_nonpiano
        pitch_hist.update(n.pitch for n in notes)
        rows.append({
            "file": path.name,
            "out": str(out_path.relative_to(cfg.output_dir)) if not cfg.dry_run else "",
            "split": split,
            "notes": len(notes),
            "dropped_nonpiano": dropped_nonpiano,
            "dup_onsets_removed": dup_removed,
            "duration_s": round(seconds, 2),
            "bpm": round(bpm, 1),
            "pitch_min": min(n.pitch for n in notes),
            "pitch_max": max(n.pitch for n in notes),
            "sha1": byte_hash,
        })

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(cfg.output_dir / "manifest.csv", rows)
    _write_csv(cfg.output_dir / "rejected.csv", rejected)

    if cfg.plot and rows:
        _plot(cfg.output_dir, pitch_hist, [r["notes"] for r in rows])

    print(f"\nKept {len(rows)} files, rejected {len(rejected)}")
    for split in ("train", "validation", "test"):
        print(f"  {split:<11} {counts[split]} files")
    print(f"  excluded as generated: {len(excluded)}")
    print(f"  total notes: {total_notes}")
    print(f"  total music: {total_seconds / 3600:.2f} hours")
    print(f"  duplicate onsets collapsed: {total_dups}")
    print(f"  non-piano notes dropped: {total_nonpiano}")
    if rejected:
        reasons = Counter(r["reason"].split(":")[0] for r in rejected)
        print("  rejection reasons:", dict(reasons.most_common(5)))
    print(f"\nManifest: {cfg.output_dir / 'manifest.csv'}")

    return {"kept": len(rows), "rejected": len(rejected), "counts": dict(counts),
            "notes": total_notes, "hours": total_seconds / 3600}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(out_dir: Path, pitch_hist: Counter, note_counts: list[int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), facecolor="#97BACB")
    axes[0].hist(list(pitch_hist.elements()), bins=range(21, 110, 2), color="#444160")
    axes[0].set(title="Pitch distribution after cleaning", xlabel="MIDI pitch", ylabel="note count")
    axes[1].hist(note_counts, bins=40, color="#444160")
    axes[1].set(title="Notes per piece", xlabel="notes", ylabel="pieces")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_overview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Overview plot: {out_dir / 'dataset_overview.png'}")


def _build_fixture(directory: Path) -> tuple[list[int], list[int]]:
    """Write a small MIDI with a piano track, a drum track and a string track."""
    directory.mkdir(parents=True, exist_ok=True)
    piano_pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
    drum_pitches = [36, 42, 36, 42]
    string_pitches = [48, 52, 55, 48]

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))
    track.append(mido.Message("program_change", program=48, channel=2, time=0))
    track.append(mido.Message("program_change", program=0, channel=9, time=0))

    track.append(mido.Message("control_change", control=64, value=127, channel=0, time=0))
    step = 240
    for pitch in piano_pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, channel=0, time=step))
    for pitch in (60, 64, 67):
        track.append(mido.Message("note_on", note=pitch, velocity=70, channel=0, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=step))
    for pitch in (60, 67):
        track.append(mido.Message("note_off", note=pitch, velocity=0, channel=0, time=0))
    track.append(mido.Message("control_change", control=64, value=0, channel=0, time=0))

    for pitch in drum_pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=100, channel=9, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, channel=9, time=step))
    for pitch in string_pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=90, channel=2, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, channel=2, time=step))

    mid.save(str(directory / "fixture.mid"))
    return piano_pitches + [60, 64, 67], drum_pitches + string_pitches


def run_self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        expected_piano, expected_other = _build_fixture(raw)

        cfg = Config(input_dir=raw, output_dir=tmp_path / "processed",
                     min_notes=1, min_duration=0.0, plot=False)
        summary = prepare(cfg)
        assert summary["kept"] == 1, summary

        written = list((tmp_path / "processed").rglob("*.mid"))
        assert len(written) == 1, written

        back = mido.MidiFile(str(written[0]))
        notes, bpm, seconds = extract_notes(back, apply_sustain=False)
        assert abs(bpm - 120.0) < 1e-6, f"initial tempo misread: {bpm}"
        assert 2.0 < seconds < 3.5, f"integrated duration implausible: {seconds}"
        pitches = [n.pitch for n in notes]

        assert set(pitches) <= set(expected_piano), f"non-piano leaked in: {sorted(set(pitches))}"
        assert not (set(pitches) & set(expected_other)), "drum/string notes survived"
        assert len(pitches) == len(expected_piano), f"expected {len(expected_piano)} notes, got {len(pitches)}"
        assert all(n.end > n.start for n in notes), "zero-length note produced"
        assert all(abs((n.start * 4) - round(n.start * 4)) < 1e-6 for n in notes), "onset off the grid"

        first = min(notes, key=lambda n: n.start)
        assert first.duration > 0.5, f"sustain pedal not applied: {first}"

        assert summary["counts"].get("train", 0) == 1, summary

        dupes = [Note(60, 1.0, 1.5, 90, 0), Note(60, 1.0, 2.0, 90, 0), Note(62, 1.0, 1.5, 90, 0)]
        collapsed, removed = collapse_duplicate_onsets(dupes)
        assert removed == 1, removed
        assert len(collapsed) == 2, collapsed
        assert max(n.duration for n in collapsed) == 1.0, "longest duplicate not kept"

        print(f"\nSELF-TEST PASSED: {len(pitches)} piano notes kept, "
              f"{len(expected_other)} non-piano notes dropped, "
              f"duration {seconds:.1f}s at {bpm:.0f} bpm")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raw", help="folder with scraped/downloaded MIDI")
    parser.add_argument("--output", default="data/processed", help="destination for the clean dataset")
    parser.add_argument("--grid", type=float, default=0.25, help="quantisation step in beats (0 disables)")
    parser.add_argument("--min-notes", type=int, default=30)
    parser.add_argument("--min-duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--pitch-low", type=int, default=21)
    parser.add_argument("--pitch-high", type=int, default=108)
    parser.add_argument("--no-sustain", action="store_true",
                        help="keep raw note lengths instead of following the pedal")
    parser.add_argument("--strict-programs", action="store_true",
                        help="drop channels that never declare a program (some files omit it)")
    parser.add_argument("--limit", type=int, default=None, help="only process N random files")
    parser.add_argument("--dry-run", action="store_true", help="report stats without writing MIDI")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                        help="skip files whose name matches this glob (repeatable)")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="also keep files produced by the melody generator")
    parser.add_argument("--self-test", action="store_true", help="verify the pipeline on a synthetic file")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    cfg = Config(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        grid=args.grid,
        min_notes=args.min_notes,
        min_duration=args.min_duration,
        pitch_low=args.pitch_low,
        pitch_high=args.pitch_high,
        apply_sustain=not args.no_sustain,
        allow_default_program=not args.strict_programs,
        limit=args.limit,
        dry_run=args.dry_run,
        plot=not args.no_plot,
        exclude=tuple(args.exclude) + (() if args.no_default_excludes else DEFAULT_EXCLUDES),
    )
    if not cfg.input_dir.exists():
        raise SystemExit(
            f"{cfg.input_dir} does not exist. Fetch data first:\n"
            "  python data/fetch_kaggle.py --dataset piano-midi-de\n"
            "  python data/fetch_kaggle.py --direct maestro-v3-midi"
        )
    prepare(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
