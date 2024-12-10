#!/usr/bin/env python3
import os
import csv
import json
import argparse
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
    "surprised": "sur",
    "neutral": "neu"
}

def process_csv(csv_path, base_dir, output_dir):
    """
    Processes the CSV file to create a JSON manifest.

    Args:
        csv_path (str): Path to the CSV file.
        base_dir (str): Base directory to prepend to filenames.
        output_dir (str): Directory where the JSON file will be saved.

    Returns:
        None
    """
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
                print(f"Invalid row {total_rows}: Emotion '{emotion_csv}' not in mapping.")
                continue

            emo_mapped = EMOTION_MAPPING[emotion_csv]

            # Get WAV file path from the 'filename' column
            wav_filename = row['filename'].strip()
            wav_path = os.path.join(base_dir, wav_filename)
            if not os.path.isfile(wav_path):
                logger.warning(f"WAV file '{wav_path}' does not exist. Skipping entry.")
                continue

            # Calculate length using WAV metadata
            try:
                with wave.open(wav_path, 'r') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    length = frames / float(rate)
            except wave.Error as e:
                logger.warning(f"Error reading WAV file '{wav_path}': {e}. Skipping entry.")
                continue

            # Construct key
            key = os.path.splitext(wav_filename)[0]

            # Add entry to JSON dictionary
            json_dict[key] = {
                "wav": os.path.abspath(wav_path),
                "length": round(length, 3),  # Rounded to milliseconds
                "emo": emo_mapped
            }
            valid_rows += 1

    # Define output JSON path
    json_filename = "output.json"
    json_path = os.path.join(output_dir, json_filename)

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_dict, json_file, indent=2)

    logger.info(f"Saved {valid_rows} valid entries to '{json_path}' out of {total_rows} total rows.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare emotion recognition dataset by creating JSON manifests from CSV and WAV files."
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Absolute path to the directory containing the CSV file.'
    )
    parser.add_argument(
        '--csv_name',
        type=str,
        required=True,
        help='Name of the CSV file.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output JSON file will be saved.'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory is set to '{args.output_dir}'.")

    # Construct full CSV path
    csv_path = os.path.join(args.base_dir, args.csv_name)
    if not os.path.isfile(csv_path):
        logger.error(f"CSV file '{csv_path}' does not exist.")
        return

    # Process the CSV
    process_csv(csv_path, args.base_dir, args.output_dir)

    logger.info("Data preparation complete.")

if __name__ == "__main__":
    main()
