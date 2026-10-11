# MobiEmotion: A Mobile-Captured Student Facial Expression Video Dataset Across High-End and Low-End Devices for Emotion Recognition Research

This repository contains the reproducible preparation code and documentation for the MobiEmotion dataset. The dataset consists of student facial-expression videos recorded on different mobile-device categories and organized for emotion-recognition research.

The video files are not included in this GitHub repository because they contain identifiable human faces and voices. Dataset release should be completed through the approved data-sharing channel after institutional, consent, and license checks are finalized.

## Dataset Summary

- Dataset name: MobiEmotion
- Modality: Mobile-captured video
- Task: Facial emotion recognition
- Emotion classes: Happy, Sad, Surprise
- Dataset size: 127 videos
- Class distribution: 50 Happy, 40 Sad, 37 Surprise
- Duration standard: 10 to 20 seconds per video after preparation
- Device categories: High-End Mobile and Low-End Mobile, classified by encoded video width
- Capture categories: Selfie and Non-Selfie, classified by orientation and filename evidence
- Split structure: Train 80%, Validation 10%, Test 10%

## Repository Contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── scripts/
    └── prepare_mobi_emotion_dataset.py
```

## Expected Input

Place the original videos in one source folder before running the script. Emotion labels are inferred from filename keywords or category folder names:

- `happy`
- `sad`
- `surprise` or `surprised`

Supported video extensions are `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`, `.3gp`, and `.mts`.

## Output Structure

The pipeline creates the following machine-learning directory structure:

```text
organized_dataset/
├── Train/
│   ├── happy/
│   ├── sad/
│   └── surprise/
├── Validation/
│   ├── happy/
│   ├── sad/
│   └── surprise/
├── Test/
│   ├── happy/
│   ├── sad/
│   └── surprise/
├── dataset_inventory.csv
└── README.md
```

Each video is copied with a standardized filename such as `happy_0001.mp4`. Files longer than 20 seconds are trimmed to the first 20 seconds. Files shorter than 10 seconds are padded by freezing the final frame until the video reaches 10 seconds.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/prepare_mobi_emotion_dataset.py --source "C:\path\to\original\videos" --output organized_dataset
```

Optional settings:

```bash
python scripts/prepare_mobi_emotion_dataset.py --source "C:\path\to\videos" --output organized_dataset --seed 42
```

## Metadata Fields

The generated `dataset_inventory.csv` includes:

- standardized relative path
- original relative path
- emotion label
- split name
- file extension
- file size
- SHA-256 checksum
- encoded and display resolution
- frame count
- frames per second
- estimated duration
- orientation
- device-class heuristic
- capture-type heuristic
- duration-adjustment action
- metadata extraction status

## Device and Capture Heuristics

Device category is assigned from encoded video width:

- High-End Mobile: encoded width greater than or equal to 1280 pixels
- Low-End Mobile: encoded width lower than 1280 pixels

Capture category is assigned as:

- Selfie: filename contains `selfie` or display orientation is portrait
- Non-Selfie: display orientation is landscape

These are dataset-preparation heuristics and should not be treated as verified device-model or camera-facing labels.

## Ethics and Access Notice

The dataset contains identifiable student facial video and may contain voice or background audio. The dataset should be shared only after confirming institutional approval, participant consent coverage for public dataset release, and the final license terms required by the target repository.

For IEEE DataPort submission, include the finalized institutional approval or exemption reference, consent statement, data-access conditions, and license terms in the dataset record and documentation.

## Citation

Use the final IEEE DataPort citation after the dataset DOI is assigned.

```text
Maneesha L L S, "MobiEmotion: A Mobile-Captured Student Facial Expression Video Dataset Across High-End and Low-End Devices for Emotion Recognition Research," IEEE DataPort, year, doi: pending.
```

## Contact

Maneesha L L S  
maneesha@rcee.ac.in
