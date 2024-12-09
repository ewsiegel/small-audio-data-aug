import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import classification_report, confusion_matrix
from dataset import IEMOCAPDataset
from model import Wav2Vec2_MLP
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_values, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_values, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=["ang", "hap", "sad", "neu"])
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm

def main():
    DATA_JSON = "data/iemocap_test.json"  # Path to the prepared JSON
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Initialize dataset and dataloader
    dataset = IEMOCAPDataset(json_path=DATA_JSON, processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = Wav2Vec2_MLP(num_classes=4, freeze_wav2vec=True)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Optionally, add validation here if you have a separate validation set

        # Save model checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # After training, evaluate on the test set
    print("Evaluating on the test set...")
    report, cm = evaluate(model, dataloader, device)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
        f.write(report)
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix.json"), 'w') as f:
        json.dump(cm.tolist(), f, indent=2)
    print(f"Saved evaluation results to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
