# Training

1. Generate json data. Refer to meld_json/meld_prepare.py and the json files in there for the expected format. MAKE SURE YOUR WAVs are monochannel.

2. Configure a hyperparams file. Refer to hparams/train_with_wav2vec2.yaml for the expected format. Make sure to load in wav2vec2 checkpoint, but don't load in mlp checkpoint if training from scratch. Details at the bottom of the yaml file.

3. Run
```bash
python train_with_wav2vec2.py PATH_TO_HYPERPARAMS_FILE
```
make sure poetry dependencies are installed and activated.

# Evaluation
