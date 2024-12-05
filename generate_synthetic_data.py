import subprocess
import uuid
from pathlib import Path
import logging
from typing import List, Dict
import random
from llm_generated_utterances import utterances

def generate_synthetic_samples(
    emotion_to_wav_files: Dict[str, List[str]],
    utterances: List[str],
    n_samples_per_wav: int = 5,
    base_output_dir: str = "synthetic_data"
) -> None:
    """
    Generate synthetic audio samples for each wav file using the provided utterances.
    
    Args:
        emotion_to_wav_files: Dictionary mapping emotion labels to lists of wav file paths
        utterances: List of utterances to use for generation
        n_samples_per_wav: Number of synthetic samples to generate per wav file
        base_output_dir: Base directory for output files
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
        
        logger.info(f"Processing emotion: {emotion}")
        
        # Process each wav file
        for wav_file in wav_files:
            logger.info(f"Processing wav file: {wav_file}")
            
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
                    
                    logger.info(f"Generated: {output_dir}/{output_filename}")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error generating sample: {e}")
                    logger.error(f"Command output: {e.output}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

# Example usage
if __name__ == "__main__":
    # Example emotion to wav files mapping
    emotion_to_wav_files = {
        "sad": ["test.wav"]
    }
    
    # Generate synthetic samples
    generate_synthetic_samples(
        emotion_to_wav_files=emotion_to_wav_files,
        utterances=utterances,
        n_samples_per_wav=5
    )