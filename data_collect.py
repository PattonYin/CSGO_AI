from mouse_tracking.mouse_tracking import collect
from video_input.screen_input import screen_capture
import threading

if __name__ == "__main__":
    t1 = threading.Thread(target=screen_capture, kwargs={'save': True, 'save_path': "data/aim_training/1/img_raw"})
    t2 = threading.Thread(target=collect, args=("data/aim_training/1/mouse_keyboard/mouse_keyboard_input.txt", ))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    