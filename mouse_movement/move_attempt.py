import pyautogui
import time
import matplotlib.pyplot as plt


def locate_head():
    coordinates = [[ 926.6871,  264.3242],
        [ 937.6829,  250.8728],
        [ 910.3350,  247.6299],
        [   0.0000,    0.0000],
        [ 872.2078,  253.4953],
        [ 970.6617,  344.2137],
        [ 809.8556,  336.8555],
        [1045.0017,  457.9584],
        [ 731.6169,  453.8897],
        [1022.3890,  454.9316],
        [ 758.9736,  444.5585],
        [ 944.4884,  570.4553],
        [ 835.4719,  570.0531],
        [ 995.8375,  755.5565],
        [ 825.5121,  762.1981],
        [   0.0000,    0.0000],
        [   0.0000,    0.0000]]
    
    # I would like the points to be plotted without lines connecting them
    plt.plot(*zip(*coordinates), marker='o', color='r', ls='')
    plt.show()    
         
         
def find_mouse_position():
    while True:
        print(pyautogui.position())
        time.sleep(1)
if __name__ == "__main__":
    # pyautogui.moveTo(840.1719, 389.6905)
    find_mouse_position()
    # locate_head()
    
#                   pyautogui                  yolo
# Left Knee         Point(x=573, y=554)        [ 926.6871,  264.3242]
# Right Knee        Point(x=685, y=557)        [ 995.8375,  755.5565]
# Head              Point(x=621, y=210)        [ 825.5121,  762.1981] 64, 347   170, -7

                    # 635, 274                    992.7006225585938, 354.89410400390625
                    # 686 274                   714, 235    -28 +39