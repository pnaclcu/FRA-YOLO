from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

model = YOLO('yolov8m-fra.yaml')

# add_swanlab_callback(
#     model,
#     project="ultralytics",
#     experiment_name="FRA-YOLO",
#     description="yolov8m on visdrone",
#
#
# )
train_results = model.train(
    data="visdrone10.yaml",  # Path to dataset configuration file
    epochs=350,  # Number of training epochs
    imgsz=640,  # Image size for training
    device='0',
    plots=True,
    batch = 4,
    amp = False,
)
metrics = model.val()



