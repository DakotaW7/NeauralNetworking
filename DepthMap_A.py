import torch
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import numpy as np
from torchvision.transforms import Compose
from PIL import Image

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
model.to(device)
model.eval()

# Set up transforms
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=True,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform the image
    img_input = transform({"image": frame_rgb})["image"]
    
    # Add batch dimension and send to device
    img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = model(img_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth = depth.cpu().numpy()
    
    # Normalize depth for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    
    # Convert to colormap
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    
    # Show both original and depth frames
    cv2.imshow('Original', frame)
    cv2.imshow('Depth', depth_color)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
