import torch
import torchvision.transforms as transforms
# from models.modeling_finetune import vit_base_patch16_224
from models.modeling_finetune import vit_large_patch16_224
from decord import VideoReader, cpu
import numpy as np

# import os
# print(f"{os.getcwd()}")

checkpoint_path = "../finetune/output/vit_large_patch16_224/checkpoint-best.pth"

def preprocess_video(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video = vr.get_batch(indices).asnumpy()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    video = torch.stack([transform(frame) for frame in video])
    video = video.permute(1, 0, 2, 3)  # Rearrange to (C, T, H, W)
    return video.unsqueeze(0)  # Add batch dimension


# Load the fine-tuned model
# model = vit_base_patch16_224(num_classes=3)  # Adjust num_classes as needed
model = vit_large_patch16_224(num_classes=3)  # Adjust num_classes as needed

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# Inference function
def predict(video_path):
    video = preprocess_video(video_path)
    with torch.no_grad():
        output = model(video)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# Test on a new video
video_path = '../finetune/inputs/videos/video_clip_000224.mp4'
predictions = predict(video_path)
print(f"Class probabilities: {predictions}")
