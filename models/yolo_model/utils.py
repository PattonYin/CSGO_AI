import numpy as np
import cv2

def txt_to_label(txt_path):
    """converst txt file to label

    Args:
        txt_path (str):
    """
    with open(txt_path, "r") as label_data:
        # Example:
        # 2 0.5094983223632786 0.7038708398470968 0.11536928675047453 0.5769169546462403
        # convert to list with these 5 values
        data = label_data.readlines()
        output = []
        for line in data:
            line = line.strip().split(" ")
            output.append([int(line[0]), float(line[1]), float(line[2]),float(line[3]), float(line[4])])
        return output
    

def draw_boxes_and_labels(image_dir, labels, class_names):
    img_copy = cv2.imread(image_dir)
    h, w, _ = img_copy.shape  # Get the dimensions of the image
    for label_i in labels:
        class_label, center_x, center_y, width, height = label_i
        
        # Convert normalized values to pixel values
        center_x_pixel = int(center_x * w)
        center_y_pixel = int(center_y * h)
        width_pixel = int(width * w)
        height_pixel = int(height * h)
        
        # Calculate top-left and bottom-right coordinates
        x_min = int(center_x_pixel - width_pixel / 2)
        y_min = int(center_y_pixel - height_pixel / 2)
        x_max = int(center_x_pixel + width_pixel / 2)
        y_max = int(center_y_pixel + height_pixel / 2)

        # Draw the rectangle
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)

        # Label
        class_index = int(class_label)
        class_detected = class_names[class_index]
        cv2.putText(img_copy, class_detected, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
    # For demonstration purposes, replace the below display code with a return or save
    cv2.imshow('Image with Bounding Boxes', img_copy)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the window after a key press
    
    # return img_copy  # or save the image if preferred


if __name__ == '__main__':
    img_path = r"X:\code\CSGO_AI\data\Player_identification\dataset_1\train\images\1_mp4-0023_jpg.rf.554cd9a71eb1e208c5e7d7feede090dd.jpg"
    label_path = r"X:\code\CSGO_AI\data\Player_identification\dataset_1\train\labels\1_mp4-0023_jpg.rf.554cd9a71eb1e208c5e7d7feede090dd.txt"
    labels = txt_to_label(label_path)
    class_names = ['CT', 'CT_head', 'T', 'T_head']
    draw_boxes_and_labels(img_path, labels, class_names)