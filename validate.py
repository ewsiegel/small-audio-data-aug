import os
import pandas as pd

def validate_directory(csv_path, wav_dir):
    # Read the CSV
    df = pd.read_csv(csv_path)
    csv_count = len(df)
    
    # Count WAV files in directory
    wav_count = len([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    
    print(f"\nValidating {wav_dir}:")
    print(f"CSV rows: {csv_count}")
    print(f"WAV files: {wav_count}")
    print("Match: ", csv_count == wav_count)
    
    if csv_count != wav_count:
        print("\nMissing or extra files:")
        csv_files = set(df['filename'])
        wav_files = set(f for f in os.listdir(wav_dir) if f.endswith('.wav'))
        missing = csv_files - wav_files
        extra = wav_files - csv_files
        if missing:
            print("Files in CSV but missing from directory:", len(missing))
            print(list(missing)[:5], "..." if len(missing) > 5 else "")
        if extra:
            print("Files in directory but missing from CSV:", len(extra))
            print(list(extra)[:5], "..." if len(extra) > 5 else "")

# Validate each directory
directories = [
    ('./train/train.csv', './train'),
    ('./test/test.csv', './test'),
    ('./eval/eval.csv', './eval'),
    ('./train_small/train_small.csv', './train_small')
]

for csv_path, wav_dir in directories:
    validate_directory(csv_path, wav_dir)