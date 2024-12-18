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

def flick_test():
    mouse = Controller()

    for i in range(3):
        x_list = []
        y_list = []
        time_list = []
        with open(f"mouse_tracking/sample/flicks/{i+1}.txt", "r") as flick_data:
            # Sample line mouse,701,580,1711094738.2393444
            # read all the x, y, and time data in one go
            data = flick_data.readlines()
            for line in data:
                line = line.strip().split(",")
                if line[0] == "mouse":                    
                    x_list.append(int(line[1]))
                    y_list.append(int(line[2]))
                    time_list.append(float(line[3]))
        
        # Target time 0.1867809295654297
        start = time.time()
        scale = 0.835
        for i in range(1, len(x_list)):
            mouse.position = (x_list[i], y_list[i])
            time.sleep((time_list[i]-time_list[i-1])*scale)        
        end = time.time()
        print(f"Actual time: {end-start}")
        print(f"Expected time: {time_list[-1]-time_list[0]}")
    
if __name__ == "__main__":
    # pynput_test()
    # pyautogui_test()
    flick_test()