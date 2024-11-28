from ultralytics import YOLO
import cv2
import torch

def capture():
    # Check and print if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the YOLOv8 pose estimation model with CUDA
    model = YOLO('yolov8m-pose.pt')  # nano model, you can use 's', 'm', or 'l' for larger models
    model.to('cuda')  # Move model to GPU
    
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Rotate frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Get the dimensions of the frame
        height, width = frame.shape[:2]
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Run YOLOv8 pose estimation on the frame using CUDA
        results = model(cropped_frame, conf=0.3, device=0)  # device=0 specifies first GPU
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
            
        # Show the annotated webcam feed
        cv2.imshow('Webcam', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

capture()
