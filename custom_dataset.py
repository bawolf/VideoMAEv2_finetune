import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path

class BreakDanceDataset(Dataset):
    def __init__(self, csv_file, video_dir, mode='train', num_frames=16):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            video_dir (string): Directory with all the videos.
            mode (string): 'train', 'val', or 'test'
            num_frames (int): Number of frames to sample from each video.
        """
        self.video_dir = Path(video_dir)
        self.annotations = pd.read_csv(csv_file)
        self.mode = mode
        self.num_frames = num_frames
        
        # Create a mapping of class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.annotations['class'].unique())}
        
        # Define transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # If we didn't get enough frames, duplicate the last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        
        # If we got too many frames, sample evenly
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames

    def __getitem__(self, idx):
        video_name = self.annotations.iloc[idx]['video_name']
        class_name = self.annotations.iloc[idx]['class']
        video_path = self.video_dir / video_name
        
        # Load video frames
        frames = self.load_video(video_path)
        
        # Apply transforms to each frame
        transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        
        # Reshape to [C, T, H, W]
        transformed_frames = transformed_frames.permute(1, 0, 2, 3)
        
        # Get label
        label = self.class_to_idx[class_name]
        
        return transformed_frames, label

# Example usage
def create_data_loaders(data_path, batch_size=32, num_workers=4):
    # Paths
    video_dir = Path(data_path) / 'videos'
    train_csv = Path(data_path) / 'train.csv'
    val_csv = Path(data_path) / 'val.csv'
    test_csv = Path(data_path) / 'test.csv'
    
    # Create datasets
    train_dataset = BreakDanceDataset(train_csv, video_dir, mode='train')
    val_dataset = BreakDanceDataset(val_csv, video_dir, mode='val')
    test_dataset = BreakDanceDataset(test_csv, video_dir, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    data_path = "/home/bawolf/workspace/break/finetune/inputs"
    train_loader, val_loader, test_loader = create_data_loaders(data_path)
    
    # Print some information
    for videos, labels in train_loader:
        print(f"Batch shape: {videos.shape}")  # Should be [batch_size, channels, frames, height, width]
        print(f"Labels shape: {labels.shape}")
        break