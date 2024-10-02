import torch
import torchvision.transforms as transforms
from models.modeling_finetune import vit_large_patch16_224
from decord import VideoReader, cpu
import numpy as np

checkpoint_path = "../finetune/output/vit_large_patch16_224/checkpoint-best.pth"
video_label_file = '../finetune/inputs/val.csv'  # Path to your file with video paths and labels

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
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities

# Main function to check accuracy
def check_accuracy(file_path):
    total_videos = 0
    correct_predictions = 0
    incorrect_predictions = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            video_path, expected_label = line.strip().split()
            expected_label = int(expected_label)

            predicted_class, probabilities = predict(video_path)
            total_videos += 1

            if predicted_class == expected_label:
                correct_predictions += 1
            else:
                incorrect_predictions.append((video_path, expected_label, predicted_class, probabilities))

    accuracy = (correct_predictions / total_videos) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_videos})")

    if incorrect_predictions:
        print("\nIncorrect Predictions:")
        for video_path, expected, predicted, probs in incorrect_predictions:
            print(f"Video: {video_path}, Expected: {expected}, Predicted: {predicted}, Probabilities: {probs}")

# Run accuracy check
check_accuracy(video_label_file)