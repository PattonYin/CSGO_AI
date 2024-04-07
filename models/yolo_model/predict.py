from ultralytics import YOLO

model = YOLO("V2.pt")
img_path = r"X:\code\CSGO_AI\video_input\screenshots\20240221-183223.png"

if __name__ == "__main__":
    metrics = model.val() 
    # results = model(img_path) 