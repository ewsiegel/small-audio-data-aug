import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2_MLP(nn.Module):
    def __init__(self, num_classes=4, freeze_wav2vec=True):
        """
        Initializes the Wav2Vec2 + MLP model.

        Args:
            num_classes (int): Number of emotion classes.
            freeze_wav2vec (bool): Whether to freeze Wav2Vec2 parameters.
        """
        super(Wav2Vec2_MLP, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        if freeze_wav2vec:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Assuming the last hidden state has 768 dimensions
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_values, attention_mask):
        # Get hidden states from Wav2Vec2
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        
        # Perform mean pooling
        pooled = hidden_states.mean(dim=1)  # Shape: [batch_size, hidden_size]
        
        # Classify
        logits = self.classifier(pooled)  # Shape: [batch_size, num_classes]
        
        return logits
