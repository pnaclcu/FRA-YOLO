from ultralytics import YOLO


model = YOLO('yolov8m-fra.yaml')

using_swanlab = False
if using_swanlab:
    from swanlab.integration.ultralytics import add_swanlab_callback
    add_swanlab_callback(
        model,
        project="ultralytics",
        experiment_name="FRA-YOLO",
        description="yolov8m on visdrone",
    )

train_results = model.train(
    data="VisDrone.yaml",  # Path to dataset configuration file, the VisDrone will be downloaded automaticall.
    epochs=350,  # Number of training epochs
    imgsz=640,  # Image size for training
    device='0',
    plots=True,
    batch = 8,
    amp = False,
)
metrics = model.val()



