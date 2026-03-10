from ultralytics import YOLO
import os, torch
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False

dataset_type = 'visdrone10' # dataset type
paths = ['./runs/detect/train2/weights/best.pt'] # model path

device = '0'
for path in paths:
    model = YOLO(path)
    metrics = model.val(data="{}.yaml".format(dataset_type), imgsz=640, batch=8, conf=0.001, iou=0.5, device=device, save=True,
                        show_boxes=True, split="val")
    print ('mAP#@75 = {}'.format(metrics.box.map75))

    metrics = model.val(data="{}.yaml".format(dataset_type), imgsz=640, batch=8, conf=0.001, iou=0.5, device=device, save=True,
                        show_boxes=True, split="test")
    print('mAP#@75 = {}'.format(metrics.box.map75))












































