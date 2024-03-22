import cv2
from ultralytics import YOLO
import time
import win32gui
import numpy as np
import pyautogui
import torch

from utils import tensor_check
from video_input.screen_input import grab_window, resolution



def run_attempt(resolution=resolution, fps=10):
    process_interval = 1/fps
    model = YOLO('pose_identification/models/YOLOv8l-pose.pt')  # load an official model
    print("model loaded")

    hwin = win32gui.FindWindow(None,'Counter-Strike 2') 
    last_time = time.time()
    # Change the name here
    while True:
        current_time = time.time()
        if current_time - last_time > process_interval:
            img_small = grab_window(hwin, game_resolution=resolution, SHOW_IMAGE=False)
            last_time = current_time
            
            if True:
                # because we use a shrunk image for input into the NN
                # we kind of want to make it larger so we can see what's happening
                # of course it's lossy compared to the original game
                target_width = 800
                scale = target_width / img_small.shape[1] # how much to magnify
                dim = (target_width,int(img_small.shape[0] * scale))
                scale=1
                dim = (int(img_small.shape[1]*scale),int(img_small.shape[0]*scale))
                resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite("video_input/temp/temp_02.jpg", resized)
                results = model(resized)
                for result in results:
                    aiming(result)
                    result.save("video_input/temp/temp_01.jpg")
                    img = cv2.imread("video_input/temp/temp_01.jpg")
                    cv2.imshow("result", img)

            # Quite loop if 'o' is pressed
            if cv2.waitKey(1) & 0xFF == ord('o'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

    time_end = time.time()
    avg_time = (time_end-time_start)/n_grabs
    fps = 1/avg_time
    print('avg_time',np.round(avg_time,5))
    print('fps',np.round(fps,2))
    return

xy_tuning = [-28, +39]

def aiming(result):
    keypoints = result.keypoints.xy
    if tensor_check.is_empty_and_matches(keypoints):
        ("No agent identified")
        return
    
    for person_keypoints in keypoints:
        count = 0
        for x in person_keypoints:
            if x[0] == 0.0 and x[1] == 0.0:
                count += 1  
        if count > 7:
            print("No agent identified")
            continue
        else:
            print("agent identified")
            x_target = person_keypoints[3][0]+xy_tuning[0]
            y_target = person_keypoints[3][1]+xy_tuning[1]
            print(f"attempting to flick to {x_target}, {y_target}")
            flick(x_target, y_target)

def flick(x,y):
    pyautogui.moveTo(x,y)
    

    

if __name__ == "__main__":
    # fps_capture_test()
    run_attempt()
    