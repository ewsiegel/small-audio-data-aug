# ########################################
# Emotion recognition from speech using wav2vec2
# For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf
# ########################################

# Seed needs to be set at top of yaml, before objects with parameters are made
seed: 6969
__set_seed: !apply:speechbrain.utils.seed_everything [!ref <seed>]

output_folder: !ref results/train_with_wav2vec2/<seed>
save_folder: !ref <output_folder>/save
train_log: !ref <output_folder>/train_log.txt

# URL for the wav2vec2 model, you can change to benchmark different models
# Important: we use wav2vec2 base and not the fine-tuned one with ASR task
# This allow you to have ~4% improvement
wav2vec2_hub: facebook/wav2vec2-base
wav2vec2_folder: !ref <save_folder>/wav2vec2_checkpoint

# different speakers for train, valid and test sets
different_speakers: False
# which speaker is used for test set, value from 1 to 10
test_spk_id: 1

# Path where data manifest files will be stored
train_annotation: /home/drew/6.7960/speechbrain/recipes/IEMOCAP/emotion_recognition/meld_json/train.json
valid_annotation: /home/drew/6.7960/speechbrain/recipes/IEMOCAP/emotion_recognition/meld_json/eval.json
test_annotation: /home/drew/6.7960/speechbrain/recipes/IEMOCAP/emotion_recognition/meld_json/test.json
split_ratio: [80, 10, 10]
skip_prep: True

# The train logger writes training statistics to a file, as well as stdout.
train_logger: !new:speechbrain.utils.train_logger.FileTrainLogger
    save_file: !ref <train_log>

ckpt_interval_minutes: 15 # save checkpoint every N min

####################### Training Parameters ####################################
number_of_epochs: 30
batch_size: 4
lr: 0.0001
lr_wav2vec2: 0.00001

#freeze all wav2vec2
freeze_wav2vec2: False
#set to true to freeze the CONV part of the wav2vec2 model
# We see an improvement of 2% with freezing CNNs
freeze_wav2vec2_conv: True

####################### Model Parameters #######################################
encoder_dim: 768

# Number of emotions
out_n_neurons: 5 # (anger, happiness, sadness, neutral, surprise`)

dataloader_options:
    batch_size: !ref <batch_size>
    shuffle: True
    num_workers: 2  # 2 on linux but 0 works on windows
    drop_last: False

# Wav2vec2 encoder
wav2vec2: !new:speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
    source: !ref <wav2vec2_hub>
    output_norm: True
    freeze: !ref <freeze_wav2vec2>
    freeze_feature_extractor: !ref <freeze_wav2vec2_conv>
    save_path: !ref <wav2vec2_folder>

avg_pool: !new:speechbrain.nnet.pooling.StatisticsPooling
    return_std: False

output_mlp: !new:speechbrain.nnet.linear.Linear
    input_size: !ref <encoder_dim>
    n_neurons: !ref <out_n_neurons>
    bias: False

epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
    limit: !ref <number_of_epochs>

modules:
    wav2vec2: !ref <wav2vec2>
    output_mlp: !ref <output_mlp>

model: !new:torch.nn.ModuleList
    - [!ref <output_mlp>]

log_softmax: !new:speechbrain.nnet.activations.Softmax
    apply_log: True

compute_cost: !name:speechbrain.nnet.losses.nll_loss

error_stats: !name:speechbrain.utils.metric_stats.MetricStats
    metric: !name:speechbrain.nnet.losses.classification_error
        reduction: batch

opt_class: !name:torch.optim.Adam
    lr: !ref <lr>

wav2vec2_opt_class: !name:torch.optim.Adam
    lr: !ref <lr_wav2vec2>

lr_annealing: !new:speechbrain.nnet.schedulers.NewBobScheduler
    initial_value: !ref <lr>
    improvement_threshold: 0.0025
    annealing_factor: 0.9
    patient: 0

lr_annealing_wav2vec2: !new:speechbrain.nnet.schedulers.NewBobScheduler
    initial_value: !ref <lr_wav2vec2>
    improvement_threshold: 0.0025
    annealing_factor: 0.9

checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
    checkpoints_dir: !ref <save_folder>
    recoverables:
        # model: !ref <model> # uncomment to load mlp from checkpoint (keep commented to train from scratch)
        wav2vec2: !ref <wav2vec2>
        # lr_annealing_output: !ref <lr_annealing> # uncomment to resume training from checkpoint
        lr_annealing_wav2vec2: !ref <lr_annealing_wav2vec2>
        counter: !ref <epoch_counter>
