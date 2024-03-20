from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Parameter section
fps = 59.98
frame_width = 1280
frame_height = 768
    

# Load a model
model = YOLO('models/YOLOv8x-pose.pt')  # load an official model

# Predict with the model
def picture_proto():
    sample_01 = r'X:\code\CSGO_AI\video_input\screenshots\20240221-184410.png'
    sample_02 = "data_0/image_01.png"
    sample_03 = "data_0/image_02.png"
    sample_04 = "data_0/image_03.png"

    samples = [sample_02, sample_03, sample_04]
    results = model(samples)  # predict on an image

    for index, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename=f'data_0/result_{samples[index][-6:-4]}.jpg')  # save to disk

def video_to_frames(path):
    # Load your video
    cap = cv2.VideoCapture(path)

    # Initialize a counter
    frame_count = 0

    while True:
        # Read frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Save frame as JPEG file
        cv2.imwrite(f'data_01/in/frame_{frame_count}.jpg', frame)
        frame_count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def video_proto(to_frames=False):
    
    video_path = "data_0/video_01.mp4"
    if to_frames:
        video_to_frames(video_path)
    print("video ready")
    frames = os.listdir("data_01/in")
    for index, frame in enumerate(tqdm(frames)):
        frame = f'data_01/in/{frame}'
        result = model(frame)
        for result in result:
            result.save(filename=f'data_01/out/frame_{index}.jpg')
                
    
if __name__ == "__main__":
    # picture_proto()
    video_proto()
    pass