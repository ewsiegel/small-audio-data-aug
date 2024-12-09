#!/usr/bin/env python3
"""Script for evaluating a fine-tuned wav2vec2 model on the IEMOCAP dataset."""

import os
import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from train_with_wav2vec2 import EmoIdBrain

def dataio_prep(hparams):
    """This function prepares the test dataset to be used in the brain class."""
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define the test dataset
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig", "emo_encoded"],
    )

    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[test_data],
        output_key="emo",
    )

    return {"test": test_data}

def evaluate_model(hparams_file, run_opts, overrides):
    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create dataset objects "test".
    datasets = dataio_prep(hparams)

    # Load the model from the checkpoint
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Print the evaluation results
    print("Test statistics:", test_stats)

if __name__ == "__main__":
    # Reading command line arguments.
    # MAKE SURE TO ADD THE CORRECT PATHS FOR THE DATA AND THE CHECKPOINT IN THE hparams yaml
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Perform evaluation
    evaluate_model(hparams_file, run_opts, overrides)