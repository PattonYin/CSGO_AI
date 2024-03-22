import time
from pynput.mouse import Button, Controller
import pyautogui

def pynput_test():
    mouse = Controller()

    for i in range(10):
        mouse.position = (5*i, 0)
        time.sleep(0.001)

def pyautogui_test():
    for i in range(10):
        start = time.time()
        pyautogui.move(5*i, 0)
        end = time.time()
        print(f"start: {start}, end: {end}, elapsed: {end-start}")


if __name__ == "__main__":
    pynput_test()
    # pyautogui_test()