from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import torch

# Parameter section
fps = 59.98
frame_width = 1280
frame_height = 768
    

# Load a model
model = YOLO('pose_identification/models/YOLOv8x-pose.pt')  # load an official model

# Predict with the model
def picture_proto():
    sample_01 = r'X:\code\CSGO_AI\video_input\screenshots\20240221-184410.png'
    sample_02 = "pose_identification/data_0/image_01.png"
    sample_03 = "pose_identification/data_0/image_02.png"
    sample_04 = "pose_identification/data_0/image_03.png"

    samples = [sample_02, sample_03, sample_04]

    start = time.time()
    results = model(samples)  # predict on an image
    end = time.time()
    print(f"Time taken: {end-start}")
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # zero_tensor = torch.tensor([0.0,0.0])
    # zero_tensor = zero_tensor.to(device)
    
    for index, result in enumerate(results):
        keypoints = result.keypoints 
        print(f"for picture {index}: ")
        for person_keypoints in keypoints.xy:
            # if the number of times that [   0.0000,    0.0000] show up is greater than 7, the person is not in the frame
            count = 0
            for x in person_keypoints:
                if x[0] == 0.0 and x[1] == 0.0:
                    count += 1  
            if count > 7:
                print("No agent identified")
                continue
            else:
                print("agent identified")
        result.save(filename=f'pose_identification/data_0/result_{samples[index][-6:-4]}.jpg')  # save to disk
        img = cv2.imread(f'pose_identification/data_0/result_{samples[index][-6:-4]}.jpg')
        cv2.imshow("result", img)

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
    picture_proto()
    pass