"""
Object detection module
Contains functions for detecting objects in images using YOLOv5
"""
import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect_objects(image, model_path="yolov5s.pt", confidence=0.25):
    """
    Detect objects in an image using YOLOv5
    
    Parameters:
    image (numpy.ndarray): Input image
    model_path (str): Path to YOLOv5 model file
    confidence (float): Confidence threshold for detection (0-1)
    
    Returns:
    tuple: (image with bounding boxes, list of detections)
    """
    # Load the model
    model = YOLO(model_path)
    
    # Run detection
    results = model(image, conf=confidence)
    
    # Get result with bounding boxes
    annotated_img = results[0].plot()
    
    # Get detection results
    detections = []
    for r in results:
        for box in r.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get confidence and class
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cls_name = r.names[cls]
            
            # Add to detections list
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'box': (int(x1), int(y1), int(x2), int(y2))
            })
    
    return annotated_img, detections

def detect_objects_from_camera(model_path="yolov5s.pt", confidence=0.25):
    """
    Detect objects from camera feed
    
    Parameters:
    model_path (str): Path to YOLOv5 model file
    confidence (float): Confidence threshold for detection (0-1)
    
    Returns:
    None (opens a window with camera feed and detections)
    """
    # Load the model
    model = YOLO(model_path)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=confidence)
        
        # Plot results on frame
        annotated_frame = results[0].plot()
        
        # Display result
        cv2.imshow("YOLOv5 Detection", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def detect_objects_from_video(video_path, model_path="yolov5s.pt", confidence=0.25):
    """
    Detect objects in a video file
    
    Parameters:
    video_path (str): Path to video file
    model_path (str): Path to YOLOv5 model file
    confidence (float): Confidence threshold for detection (0-1)
    
    Returns:
    None (opens a window with video and detections)
    """
    # Load the model
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=confidence)
        
        # Plot results on frame
        annotated_frame = results[0].plot()
        
        # Display result
        cv2.imshow("YOLOv5 Detection", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
