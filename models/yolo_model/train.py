from ultralytics import YOLO


if __name__ == "__main__":
    # model = YOLO("V2_train.pt")
    # results = model.train(data="config.yaml", epochs=400, batch=12)
    
    # model = YOLO("V1.pt")
    # results = model.train(data="config.yaml", name="tuning_01", lr0=0.03, epochs=5, batch=12) 
    
    # model = YOLO("V1.pt")
    # results = model.train(data="config.yaml", name="tuning_02", epochs=5, batch=12) 
    
    model = YOLO("V1.pt")
    results = model.train(data="config.yaml", name="tuning_03", optimizer='Adam', lr0=0.03, epochs=5, batch=12)
    
# do this cd .\models\yolo_model\