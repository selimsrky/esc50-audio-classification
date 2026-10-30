import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import download_and_extract_esc50, ESC50Dataset
from model import AudioCNN

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Target compute device: {device}")

    # 1. Download and verify dataset
    download_and_extract_esc50()
    csv_path = "data/ESC-50-master/meta/esc50.csv"
    audio_dir = "data/ESC-50-master/audio"
    
    meta_df = pd.read_csv(csv_path)

    # Standard ESC-50 5-fold cross validation split (Folds 1-4 train, Fold 5 test)
    train_df = meta_df[meta_df['fold'] != 5]
    test_df = meta_df[meta_df['fold'] == 5]

    train_dataset = ESC50Dataset(train_df, audio_dir)
    test_dataset = ESC50Dataset(test_df, audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = AudioCNN(num_classes=50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 10
    print(f"\nTraining pipeline started for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = (correct / total) * 100
        avg_loss = total_loss / total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc:.2f}%")

    # 4. Evaluation on Test Split (Fold 5)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for specs, labels in test_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
    final_acc = (test_correct / test_total) * 100
    print(f"\nFinal Test Accuracy (Fold 5 Evaluation): {final_acc:.2f}%")

if __name__ == "__main__":
    main()
