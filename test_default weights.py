import torch
from models.modeling_finetune import vit_giant_patch14_224

def test_model():
    # Create a model instance
    model = vit_giant_patch14_224(num_classes=400)
    
    # Load pre-trained weights
    checkpoint = torch.load('../finetune/weights/vit_g_hybrid_pt_1200e.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a dummy input (adjust dimensions as needed)
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    
    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print("Model test successful!")

if __name__ == "__main__":
    test_model()
