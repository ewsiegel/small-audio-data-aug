#!/usr/bin/env python3
"""
Script to prepare a custom emotion recognition dataset by converting CSV annotations and corresponding .wav files into JSON manifests.

The dataset is organized into three splits: train, test, and eval. Each split contains a folder with .wav files and a CSV file describing the utterances.

Usage:
    python meld_prepare.py \
        --train_path /path/to/train_folder \
        --test_path /path/to/test_folder \
        --eval_path /path/to/eval_folder \
        --output_dir /path/to/output/jsons
"""

import os
import csv
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
import wave

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mapping from CSV emotion to JSON emotion
EMOTION_MAPPING = {
    "joy": "hap",
    "anger": "ang",
    "sadness": "sad",
    "surprise": "sur",
    "neutral": "neu"
}

def process_split(split_path, split_type, output_dir):
    """
    Processes a single data split (train, test, eval) to create a JSON manifest.

    Args:
        split_path (str): Path to the split folder containing .wav files and a CSV.
        split_type (str): Type of the split ('train', 'test', 'eval').
        output_dir (str): Directory where the JSON file will be saved.

    Returns:
        None
    """
    prefix_dict = {
        "train": "tr",
        "test": "te",
        "eval": "ev"
    }

    if split_type not in prefix_dict:
        logger.error(f"Invalid split type '{split_type}'. Must be one of 'train', 'test', 'eval'.")
        return

    prefix = prefix_dict[split_type]
    logger.info(f"Processing '{split_type}' split with prefix '{prefix}'.")

    # Identify CSV file in the split directory
    csv_files = [f for f in os.listdir(split_path) if f.endswith('.csv')]
    if not csv_files:
        logger.error(f"No CSV file found in '{split_path}'. Skipping this split.")
        return
    csv_path = os.path.join(split_path, csv_files[0])
    logger.info(f"Using CSV file: {csv_path}")

    # Identify WAV directory (assuming all .wav files are in split_path)
    wav_dir = split_path
    logger.info(f"Assuming WAV files are located in: {wav_dir}")

    json_dict = {}
    total_rows = 0
    valid_rows = 0

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1
            emotion_csv = row['Emotion'].strip().lower()

            if emotion_csv not in EMOTION_MAPPING:
                logger.debug(f"Skipping row {total_rows}: Emotion '{emotion_csv}' not in mapping.")
                continue

            emo_mapped = EMOTION_MAPPING[emotion_csv]
            dialogue_id = row['Dialogue_ID'].strip()
            utterance_id = row['Utterance_ID'].strip()

            # Construct key
            key = f"{prefix}_{dialogue_id}_{utterance_id}"

            # Get WAV file path from the 'filename' column
            wav_filename = row['filename'].strip()
            wav_path = os.path.join(wav_dir, wav_filename)
            if not os.path.isfile(wav_path):
                logger.warning(f"WAV file '{wav_path}' does not exist. Skipping entry '{key}'.")
                continue

            # Calculate length using WAV metadata
            try:
                with wave.open(wav_path, 'r') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    length = frames / float(rate)
            except wave.Error as e:
                logger.warning(f"Error reading WAV file '{wav_path}': {e}. Skipping entry '{key}'.")
                continue

            # Add entry to JSON dictionary
            json_dict[key] = {
                "wav": os.path.abspath(wav_path),
                "length": round(length, 3),  # Rounded to milliseconds
                "emo": emo_mapped
            }
            valid_rows += 1

    # Define output JSON path
    json_filename = f"{split_type}.json"
    json_path = os.path.join(output_dir, json_filename)

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_dict, json_file, indent=2)

    logger.info(f"Saved {valid_rows} valid entries to '{json_path}' out of {total_rows} total rows.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare custom emotion recognition dataset by creating JSON manifests from CSV and WAV files."
    )
    parser.add_argument(
        '--train_path',
        type=str,
        required=True,
        help='Path to the training split folder.'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        required=True,
        help='Path to the testing split folder.'
    )
    parser.add_argument(
        '--eval_path',
        type=str,
        required=True,
        help='Path to the evaluation split folder.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output JSON files will be saved.'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory is set to '{args.output_dir}'.")

    # Process each split
    splits = {
        "train": args.train_path,
        "test": args.test_path,
        "eval": args.eval_path
    }

    for split_type, split_path in splits.items():
        if not os.path.isdir(split_path):
            logger.error(f"Split path '{split_path}' does not exist or is not a directory. Skipping '{split_type}' split.")
            continue
        process_split(split_path, split_type, args.output_dir)

    logger.info("Data preparation complete.")

if __name__ == "__main__":
    main()

