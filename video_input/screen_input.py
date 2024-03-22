# adapted from: https://github.com/Sentdex/pygta5/blob/master/grabscreen.py
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pyautogui

import sys, os
import mss

sys.path.append('.')

# from config import *
from config import csgo_img_dimension, IS_CONTRAST

resolution=(1280,720)

def grab_window(hwin, game_resolution=(1024,768), SHOW_IMAGE=False):
    '''
    -- Inputs --

    hwin
    this is the HWND id of the cs go window
    we play in windowed rather than full screen mode
    e.g. https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow

    game_resolution=(1024,768)
    is the windowed resolution of the game
    I think could get away with game_resolution=(640,480)
    and should be quicker to grab from
    but for now, during development, I like to see the game in reasonable
    size

    SHOW_IMAGE
    whether to display the image. probably a bad idea to 
    do that here except for testing
    better to use cv2.imshow('img',img) outside the funcion

    -- Outputs --
    currently this function returns img_small
    img is the raw capture image, in BGR
    img_small is a low res image, with the thought of
    using this as input to a NN

    '''

    # we used to try to get the resolution automatically
    # but this didn't seem that reliable for some reason
    # left,top,right,bottom = win32gui.GetWindowRect(hwin)
    # width = right - left 
    # height = bottom - top

    # ----------Margins and Capture Area Definition----------
    # Defining margins to ignore parts of the window for efficiency.
    bar_height = 35 # height of header bar
    # how much of top and bottom of image to ignore
    # (reasoning we don't need the entire screen)
    # much more efficient not to grab it in the first place
    # rather than crop out later
    # this stage can be a bit of a bottle neck
    offset_height_top = 0 
    offset_height_bottom = 130 
    offset_sides = 0 # ignore this many pixels on sides, 
    width = game_resolution[0] - 2*offset_sides
    height = game_resolution[1]-offset_height_top-offset_height_bottom

    # ----------Capturing the Window----------
    # Capturing the specified window area using win32gui and win32ui functions.
    hwindc = win32gui.GetWindowDC(hwin) # retrieves the device context (DC) for the specified window (identified by its handle hwin). It allows you to draw directly onto the window or, in this case, capture its contents.
    srcdc = win32ui.CreateDCFromHandle(hwindc) # This creates a device context object (srcdc) from the handle obtained in the previous step. This object provides a higher-level object-oriented interface to the DC, making it easier to work with in Python.
    memdc = srcdc.CreateCompatibleDC() # It's used to create an in-memory copy of the window's content, which can then be manipulated or saved without affecting the original window.
    bmp = win32ui.CreateBitmap() # Initializes a new bitmap object. A bitmap is used to store an image in memory. This image can be drawn in the memory DC created in the previous step.
    bmp.CreateCompatibleBitmap(srcdc, width, height) # This compatibility ensures that the bitmap can be used with the memory DC to capture the window's contents. The width and height define the size of the bitmap and, consequently, the size of the image to be captured.
    memdc.SelectObject(bmp) # This step is essential for copying the window's content onto the bitmap.

    memdc.BitBlt((0, 0), (width, height), srcdc, (offset_sides, bar_height+offset_height_top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if IS_CONTRAST:
        contrast = 1.5
        brightness = 1.0
        img = cv2.addWeighted(img, contrast, img, 0, brightness)
    

    img_small = cv2.resize(img, csgo_img_dimension[::-1])


    if SHOW_IMAGE:

        target_width = 800
        scale = target_width / img_small.shape[1] # how much to magnify
        dim = (target_width,int(img_small.shape[0] * scale))
        resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('resized',resized) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return img_small
    # return img, img_small

def fps_capture_test():

    if False:
        # can use this to manually find hwin, id of selected window
        # actually can look this up from name directly
        while True:
            hwin = win32gui.GetForegroundWindow()
            print(hwin)
            time.sleep(0.2)

    time_start = time.time()
    n_grabs=20000
    folder_path = "video_input/screenshots/"
    hwin = win32gui.FindWindow(None,'Counter-Strike 2') # Change the name here
    for i in range(n_grabs):
        img_small = grab_window(hwin, game_resolution=(1024,768), SHOW_IMAGE=False)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(folder_path, f"{timestamp}.png")
        cv2.imwrite(filename, img_small)
        print(f"saved {filename}")

        if True:
            # because we use a shrunk image for input into the NN
            # we kind of want to make it larger so we can see what's happening
            # of course it's lossy compared to the original game
            target_width = 800
            scale = target_width / img_small.shape[1] # how much to magnify
            dim = (target_width,int(img_small.shape[0] * scale))
            resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('resized',resized) 
            
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


if __name__ == "__main__":
    fps_capture_test()
    


