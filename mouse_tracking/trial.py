from pynput import mouse, keyboard
import time

# This flag is used to control the termination of both listeners
stop_listeners = False

# Flags to control the state of recording
is_recording = False

# Define the start and stop keys
start_key = keyboard.Key.f2  # Start recording when F2 is pressed
stop_key = keyboard.Key.f3   # Stop recording when F3 is pressed

def on_move(x, y):
    print(f"Mouse moved to ({x}, {y}) at timestamp {time.time()}")
    if is_recording:
        # Save the coordinates to a file
        with open("mouse_tracking.txt", "a") as file:
            file.write(f"{x},{y},{time.time()}\n")

def on_click(x, y, button, pressed):
    if is_recording:
        action = "pressed" if pressed else "released"
        print(f"Mouse {action} at ({x}, {y}) with {button} at timestamp {time.time()}")

def on_press(key):
    global is_recording
    if key == start_key:
        is_recording = True
        print("Started recording data...")
    elif key == stop_key:
        is_recording = False
        print("Stopped recording data...")
        # Optionally, stop listeners if needed
        # return False

if __name__ == "__main__":
    # Start the mouse listener to handle mouse events
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    mouse_listener.start()

    # Start the keyboard listener to handle keyboard events
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    # Wait for the flag to become True to stop both listeners
    try:
        while not stop_listeners:
            time.sleep(0.1)  # Small delay to prevent busy waiting
    except KeyboardInterrupt:
        pass  # Allow manual interruption with CTRL+C

    # Stop the listeners
    mouse_listener.stop()
    keyboard_listener.stop()
