from ultralytics import YOLO

# Load a model
model = YOLO('models/yolov8x.pt')  # pretrained YOLOv8n model
frame_ids = [2057, 2073, 2305, 2325, 2405, 2489, 2524, 2544, 2589, 2689, 2850]
samples = [f'data_01/in/frame_{id}.jpg' for id in frame_ids]

# Run batched inference on a list of images
results = model(samples)  # return a list of Results objects

# Process results list
for index, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename=f'data_02/out/frame_{frame_ids[index]}.jpg')  # save to disk