import subprocess
import uuid
from pathlib import Path
import logging
from typing import List, Dict
import random
from tqdm import tqdm
from llm_generated_utterances import utterances
from emotion_to_wav import get_emotion_to_wav

def generate_synthetic_samples(
    emotion_to_wav_files: Dict[str, List[str]],
    utterances: List[str],
    n_samples_per_wav: int = 5,
    base_output_dir: str = "synthetic_data"
) -> None:
    """
    Generate synthetic audio samples for each wav file using the provided utterances.
    Shows progress bar for each emotion being processed.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create base output directory
    Path(base_output_dir).mkdir(exist_ok=True)
    
    # Process each emotion and its wav files
    for emotion, wav_files in emotion_to_wav_files.items():
        # Create emotion-specific output directory
        output_dir = Path(base_output_dir) / emotion
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nProcessing emotion: {emotion}")
        
        # Calculate total operations for this emotion
        total_operations = len(wav_files) * n_samples_per_wav
        
        # Create progress bar for this emotion
        with tqdm(total=total_operations, desc=f"{emotion}", unit='sample') as pbar:
            # Process each wav file
            for wav_file in wav_files:
                # Generate n samples for this wav file
                for _ in range(n_samples_per_wav):
                    # Generate unique filename for output
                    output_filename = f"synthetic_{uuid.uuid4()}.wav"
                    
                    # Randomly select an utterance
                    utterance = random.choice(utterances)
                    
                    try:
                        # Run the synthetic generation script
                        cmd = [
                            "bash",
                            "./run_synthetic.sh",
                            wav_file,
                            utterance,
                            str(output_dir),
                            output_filename
                        ]
                        
                        # Execute the command
                        result = subprocess.run(
                            cmd,
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        
                        # Update progress bar
                        pbar.update(1)
                        
                    except subprocess.CalledProcessError as e:
                        logger.error(f"\nError generating sample: {e}")
                        logger.error(f"Command output: {e.output}")
                        pbar.update(1)  # Still update progress even on error
                    except Exception as e:
                        logger.error(f"\nUnexpected error: {e}")
                        pbar.update(1)  # Still update progress even on error

# Example usage
if __name__ == "__main__":
    # Example emotion to wav files mapping
    emotion_to_wav_files = get_emotion_to_wav()
    del emotion_to_wav_files["neutral"]
    del emotion_to_wav_files['joy']
    del emotion_to_wav_files['sadness']
    emotion_to_wav_files['anger'] = emotion_to_wav_files['anger'][142:]
    #del emotion_to_wav_files['anger']
    #del emotion_to_wav_files['surprise']

    # Generate synthetic samples
    generate_synthetic_samples(
        emotion_to_wav_files=emotion_to_wav_files,
        utterances=utterances,
        n_samples_per_wav=5
    )
