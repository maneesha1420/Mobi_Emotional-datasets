"""Prepare the MobiEmotion dataset for machine-learning experiments.

The script scans a source folder, infers emotion labels from filenames or
category folder names, creates an 80/10/10 split, standardizes filenames,
normalizes video duration to 10-20 seconds, and writes dataset_inventory.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg


EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp", ".mts"}
CLASSES = ("happy", "sad", "surprise")
SPLITS = ("Train", "Validation", "Test")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_emotion(path: Path) -> str | None:
    text = f"{path.parent.name} {path.stem}".lower()
    matches = set(re.findall(r"happy|sad|surpris(?:e|ed)", text))
    normalized = {"surprise" if item.startswith("surpris") else item for item in matches}
    return normalized.pop() if len(normalized) == 1 else None


def scan_videos(source: Path) -> list[Path]:
    videos = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
    return sorted(videos, key=lambda p: str(p.relative_to(source)).lower())


def split_counts(total: int) -> dict[str, int]:
    train = math.floor(total * 0.8)
    validation = round(total * 0.1)
    test = total - train - validation
    return {"Train": train, "Validation": validation, "Test": test}


def build_plan(source: Path, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[Path]] = {label: [] for label in CLASSES}
    for path in scan_videos(source):
        label = infer_emotion(path)
        if label in grouped:
            grouped[label].append(path)

    plan: list[dict[str, str]] = []
    rng = random.Random(seed)
    for label in CLASSES:
        files = grouped[label]
        rng.shuffle(files)
        counts = split_counts(len(files))
        cursor = 0
        class_index = 1
        for split in SPLITS:
            for path in files[cursor : cursor + counts[split]]:
                extension = ".mp4"
                filename = f"{label}_{class_index:04d}{extension}"
                plan.append(
                    {
                        "split": split,
                        "emotion": label,
                        "original_relative_path": str(path.relative_to(source)),
                        "relative_path": str(Path(split) / label / filename),
                    }
                )
                class_index += 1
            cursor += counts[split]
    return sorted(plan, key=lambda row: (row["split"], row["emotion"], row["relative_path"]))


def video_metadata(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if min(width, height, frames) <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        display_width, display_height = (height, width) if round(rotation) % 180 else (width, height)
        duration = frames / fps
        portrait = display_height > display_width
        selfie_keyword = "selfie" in path.stem.lower()
        return {
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "encoded_width_px": width,
            "encoded_height_px": height,
            "rotation_metadata_degrees": rotation,
            "display_width_px": display_width,
            "display_height_px": display_height,
            "orientation": "Portrait" if portrait else "Landscape",
            "fps_reported": round(fps, 6),
            "frame_count_reported": frames,
            "duration_seconds_estimated": round(duration, 6),
            "device_class_heuristic": "High-End Mobile" if width >= 1280 else "Low-End Mobile",
            "capture_type_heuristic": "Selfie" if selfie_keyword or portrait else "Non-Selfie",
            "capture_type_basis": "filename contains selfie" if selfie_keyword else ("portrait display ratio" if portrait else "landscape display ratio"),
            "metadata_status": "ok",
            "metadata_error": "",
        }
    except Exception as exc:
        return {
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata_status": "error",
            "metadata_error": str(exc),
        }
    finally:
        cap.release()


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])


def normalize_video(source: Path, destination: Path, duration: float) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".partial.mp4")
    if duration > 20:
        action = "trim_first_20_seconds"
        command = [
            FFMPEG,
            "-y",
            "-i",
            str(source),
            "-t",
            "20",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temp),
        ]
        run_ffmpeg(command)
    elif duration < 10:
        action = "freeze_last_frame_to_10_seconds"
        pad_seconds = max(0.0, 10.0 - duration)
        command = [
            FFMPEG,
            "-y",
            "-i",
            str(source),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={pad_seconds:.6f}",
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temp),
        ]
        run_ffmpeg(command)
    else:
        action = "unchanged_copy"
        shutil.copy2(source, temp)
    temp.replace(destination)
    return action


def write_readme(output: Path) -> None:
    readme = output / "README.md"
    readme.write_text(
        "# MobiEmotion Dataset\n\n"
        "This folder contains the organized MobiEmotion dataset for emotion-recognition research.\n\n"
        "Structure: Train, Validation, and Test splits, each containing happy, sad, and surprise folders.\n\n"
        "See dataset_inventory.csv for file-level metadata, checksums, resolution information, duration details, "
        "device-class heuristics, and capture-type heuristics.\n",
        encoding="utf-8",
    )


def write_inventory(output: Path, rows: list[dict[str, object]]) -> None:
    inventory = output / "dataset_inventory.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MobiEmotion dataset.")
    parser.add_argument("--source", required=True, type=Path, help="Folder containing original videos.")
    parser.add_argument("--output", default=Path("organized_dataset"), type=Path, help="Output dataset folder.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for deterministic split assignment.")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    if output.exists():
        raise SystemExit(f"Output folder already exists. Move or remove it first: {output}")

    plan = build_plan(source, args.seed)
    counts = Counter(row["emotion"] for row in plan)
    missing = [label for label in CLASSES if counts[label] == 0]
    if missing:
        raise SystemExit(f"No videos found for required classes: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for index, row in enumerate(plan, 1):
        original = source / row["original_relative_path"]
        metadata = video_metadata(original)
        destination = output / row["relative_path"]
        duration = float(metadata.get("duration_seconds_estimated", 0) or 0)
        action = normalize_video(original, destination, duration)
        final_metadata = video_metadata(destination)
        records.append(
            {
                **row,
                "source_extension": original.suffix.lower(),
                "standardized_extension": destination.suffix.lower(),
                "duration_adjustment_action": action,
                "source_sha256": metadata.get("sha256", ""),
                **{f"prepared_{key}": value for key, value in final_metadata.items()},
            }
        )
        print(f"[{index:03d}/{len(plan):03d}] {row['relative_path']}")

    write_inventory(output, records)
    write_readme(output)
    (output / "preparation_summary.json").write_text(
        json.dumps({"total_videos": len(records), "class_counts": dict(counts), "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared dataset written to: {output}")


if __name__ == "__main__":
    main()
