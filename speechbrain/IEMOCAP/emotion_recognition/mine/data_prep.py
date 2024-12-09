import os
import json
import glob
from collections import defaultdict

def prepare_iemocap_data(iemocap_dir, output_json):
    """
    Processes the IEMOCAP dataset to extract audio file paths and emotion labels.

    Args:
        iemocap_dir (str): Path to the IEMOCAP_full_release directory.
        output_json (str): Path to save the processed data as JSON.
    """
    sessions = [f"Session{n}" for n in range(1, 6)]
    emotions_of_interest = {"ang": "ang", "hap": "hap", "sad": "sad", "neu": "neu"}

    data = defaultdict(list)

    for session in sessions:
        dialog_dir = os.path.join(iemocap_dir, session, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(iemocap_dir, session, "sentences", "wav")

        emo_files = glob.glob(os.path.join(dialog_dir, "*.txt"))
        for emo_file in emo_files:
            with open(emo_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        emotion = parts[4].lower()
                        if emotion in emotions_of_interest:
                            utt_id = parts[1]
                            wav_subdir = utt_id[:15]
                            wav_file = os.path.join(wav_dir, wav_subdir, f"{utt_id}.wav")
                            label = emotions_of_interest[emotion]
                            if os.path.exists(wav_file):
                                data['test'].append({
                                    "id": utt_id,
                                    "wav": wav_file,
                                    "emo": label
                                })
                            else:
                                print(f"Warning: {wav_file} does not exist.")
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data preparation complete. Saved to {output_json}")

if __name__ == "__main__":
    # Example usage
    IEMOCAP_DIR = "/home/drew/6.7960/IEMOCAP_full_release"  # Update this path
    OUTPUT_JSON = "data/iemocap_test.json"
    os.makedirs("data", exist_ok=True)
    prepare_iemocap_data(IEMOCAP_DIR, OUTPUT_JSON)
