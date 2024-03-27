import numpy as np
import cv2
import time
import threading

import win32gui
from pynput import keyboard, mouse

from ultralytics import YOLO
import torch

from utils import tensor_check
from video_input.screen_input import grab_window, resolution

xy_tuning_01 = [-20, +40]
xy_tuning_02 = [1, 1]
fps = 10
model_path = 'pose_identification/models/YOLOv8l-pose.pt'
time_stamp = time.time()
save_path_img_raw = "data/aim_training/1/img_raw"

# ----------------------------------- # 
# ----------------------------------- # 
# ------Screen Capture Section ------ # 
# ----------------------------------- # 
# ----------------------------------- # 
def screen_capture(resolution=resolution, fps=fps, save=False, save_path=save_path_img):
    process_interval = 1/fps
    hwin = win32gui.FindWindow(None,'Counter-Strike 2') 
    last_time = time.time()
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
                time_stamp = time.time()
                cv2.imshow("result", resized)
                cv2.imwrite("video_input/temp/temp_02.jpg", resized)
                if save:
                    cv2.imwrite(f"{save_path}/{time_stamp}.jpg", resized)
                
            # Quite loop if 'o' is pressed
            if cv2.waitKey(1) & 0xFF == ord('o'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    return

# ----------------------------------- # 
# ----------------------------------- # 
# ----------Aiming Section ---------- # 
# ----------------------------------- # 
# ----------------------------------- # 
def aim_at_target(x_target, y_target):
    mouse.position = (x_target, y_target)

def identify_target(result):
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
            x_target = person_keypoints[3][0]+xy_tuning_01[0]
            y_target = person_keypoints[3][1]+xy_tuning_01[1]
            print(f"attempting to flick to {x_target}, {y_target}")
            aim_at_target(x_target, y_target)

def on_press_flick(key):
    try:
        if key.char == 'f':  # Check if the pressed key is 'f'
            img = cv2.imread("video_input/temp/temp_02.jpg")
            results = model(img)
            for result in results:
                identify_target(result)
    except AttributeError:
        print(f'Special key {key} pressed')

def on_release(key):
    # Stop listener
    if key == keyboard.Key.esc:
        return False    
        
def auto_aim():
    """
    If the button 'f' is pressed, the program will analyze the more recent img and attempt to flick to the target
    """
    with keyboard.Listener(
            on_press=on_press_flick,
            on_release=on_release) as listener:
        listener.join()
        
# ----------------------------------- # 
# ----------------------------------- # 
# ------Load model & threading ------ # 
# ----------------------------------- # 
# ----------------------------------- # 
mouse = mouse.Controller()
model = YOLO(model_path)
print("model loaded")

def run():
    t1 = threading.Thread(target=screen_capture)
    t2 = threading.Thread(target=auto_aim)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
if __name__ == "__main__":
    run()