import torch
import torchvision.transforms as transforms
from models.modeling_finetune import vit_large_patch16_224
from decord import VideoReader, cpu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import json
import glob
import os

class BreakDanceDataset(Dataset):
    def __init__(self, csv_file, num_frames=16):
        self.data = pd.read_csv(csv_file, header=None, sep=' ')
        self.video_paths = self.data[0].values
        self.labels = self.data[1].values
        self.num_frames = num_frames
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        video = vr.get_batch(indices).asnumpy()
        
        # Transform frames
        video = torch.stack([self.transform(frame) for frame in video])
        video = video.permute(1, 0, 2, 3)  # Rearrange to (C, T, H, W)
        
        return video, label

def plot_training_curves(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "log.txt"))
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for log_file in log_files:
        print(f"Reading log file: {log_file}")
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    print(f"Log line keys: {data.keys()}")
                    if 'train_loss' in data:
                        train_loss.append(data['train_loss'])
                        train_acc.append(data.get('train_acc1', 0))
                    if 'test_loss' in data:
                        val_loss.append(data['test_loss'])
                        val_acc.append(data.get('test_acc1', 0))
                except json.JSONDecodeError:
                    print(f"Could not parse line: {line}")
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    print(f"Collected data points: Train Loss: {len(train_loss)}, Val Loss: {len(val_loss)}")
    print(f"Collected data points: Train Acc: {len(train_acc)}, Val Acc: {len(val_acc)}")
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'))
    plt.close()

def generate_confusion_matrix(model, data_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))
    plt.close()
    
    return cm

if __name__ == "__main__":
    # Paths
    log_dir = "/home/bawolf/workspace/break/finetune/output/vit_large_patch16_224/"
    checkpoint_path = os.path.join(log_dir, "checkpoint-best.pth")
    test_csv = "/home/bawolf/workspace/break/finetune/inputs/test.csv"
    
    # Plot training curves
    plot_training_curves(log_dir)
    
    # Load model
    model = vit_large_patch16_224(num_classes=3)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test dataset and dataloader
    test_dataset = BreakDanceDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(model, test_loader, device)
    
    print("Visualization complete! Check the output directory for plots.")