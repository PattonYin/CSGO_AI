import win32gui

# Define the constant if it's not available
if not hasattr(win32gui, 'GW_HWNDNEXT'):
    GW_HWNDNEXT = 2
else:
    GW_HWNDNEXT = win32gui.GW_HWNDNEXT

def find_cs2_window():
    handle = win32gui.FindWindow(None, None)  # Find any window
    while handle:
        title = win32gui.GetWindowText(handle)
        if "Counter-Strike 2" in title:
            return handle, title
        handle = win32gui.GetWindow(handle, GW_HWNDNEXT)  # Move to the next window
    return None, None

# Example usage:
handle, title = find_cs2_window()
if handle:
    print("Counter-Strike 2 window found!")
    print("Window handle:", handle)
    print("Window title:", title)
else:
    print("Counter-Strike 2 window not found.")
