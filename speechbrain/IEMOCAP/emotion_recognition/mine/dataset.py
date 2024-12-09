import json
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
import librosa

class IEMOCAPDataset(Dataset):
    def __init__(self, json_path, processor):
        """
        Initializes the IEMOCAP dataset.

        Args:
            json_path (str): Path to the JSON file containing data.
            processor (Wav2Vec2Processor): Hugging Face Wav2Vec2 processor.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)['test']
        
        self.processor = processor
        self.label_mapping = {"ang": 0, "hap":1, "sad":2, "neu":3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        wav_path = sample['wav']
        label = self.label_mapping[sample['emo']]

        # Load audio
        waveform, sr = librosa.load(wav_path, sr=16000)
        
        # Process audio
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Squeeze to remove extra dimension
        input_values = inputs.input_values.squeeze()  # Shape: [sequence_length]
        attention_mask = inputs.attention_mask.squeeze()  # Shape: [sequence_length]

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }
