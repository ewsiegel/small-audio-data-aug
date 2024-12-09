"""
Test script for emotion recognition model on IEMOCAP dataset.
"""

import os
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.checkpoints import Checkpointer

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform to the output probabilities."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs, lens)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            predictions = torch.argmax(predictions, dim=-1)
        return predictions

def evaluate_model(hparams_file, data_folder=None, checkpoint_path=None):
    """
    Evaluate the model on IEMOCAP test set.
    """

    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Override data_folder if provided
    if data_folder is not None:
        hparams["data_folder"] = data_folder

    # Create brain class instance
    emotion_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    # Initialize checkpointer
    checkpointer = Checkpointer(
        os.path.dirname(checkpoint_path),
        recoverables={
            "wav2vec2": emotion_brain.modules.wav2vec2,
            "model": emotion_brain.modules.output_mlp,
        }
    )

    # Load checkpoint using recover_if_possible instead
    checkpointer.recover_if_possible()

    # Prepare the data if not already done
    from iemocap_prepare import prepare_data
    run_on_main(
        prepare_data,
        kwargs={
            "data_original": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": hparams["split_ratio"],
            "different_speakers": hparams["different_speakers"],
            "test_spk_id": hparams["test_spk_id"],
        },
    )

    # Load test set
    datasets = dataio_prep(hparams)
    test_set = datasets["test"]

    # Create test dataloader
    test_loader = sb.dataio.dataloader.make_dataloader(
        test_set, **hparams["dataloader_options"]
    )

    # Initialize confusion matrix
    emotion_labels = ["ang", "hap", "sad", "neu"]  # Update if you have different emotions
    confusion_matrix = torch.zeros(len(emotion_labels), len(emotion_labels))

    # Testing
    emotion_brain.modules.eval()
    print("Starting evaluation...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Get predictions
            predictions = emotion_brain.evaluate_batch(batch, stage=sb.Stage.TEST)

            # Get true labels
            true_labels = batch.emo_encoded.data

            # Update confusion matrix
            for pred, true in zip(predictions, true_labels):
                confusion_matrix[true, pred] += 1

    # Calculate metrics
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)

    # Print results
    print("\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-class Accuracy:")
    for label, acc in zip(emotion_labels, per_class_accuracy):
        print(f"{label}: {acc:.4f}")

    print("\nConfusion Matrix:")
    print("True\\Pred", end="\t")
    for label in emotion_labels:
        print(f"{label}", end="\t")
    print()

    for i, true_label in enumerate(emotion_labels):
        print(f"{true_label}", end="\t")
        for j in range(len(emotion_labels)):
            print(f"{confusion_matrix[i,j]:.0f}", end="\t")
        print()

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = emotion_brain.hparams.label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo", "emo_encoded"],
        )
    return datasets

if __name__ == "__main__":
    # Example usage
    hparams_file = "hparams/train_with_wav2vec2.yaml"
    checkpoint_path = "/home/drew/6.7960/speechbrain/recipes/IEMOCAP/emotion_recognition/results/train_with_wav2vec2/1993/save/CKPT+2024-12-04+22-24-05+00"  # Update with your checkpoint path
    data_folder = "/home/drew/6.7960/IEMOCAP_full_release"  # Update with your IEMOCAP path
    
    evaluate_model(
        hparams_file=hparams_file,
        data_folder=data_folder,
        checkpoint_path=checkpoint_path
    )
