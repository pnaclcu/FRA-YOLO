from ultralytics import YOLO
import os, torch
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False

dataset_type = './t_drone_person' # dataset type
epochs = [81]
paths = ['./runs/detect/train{}/weights/best.pt'.format(e) for e in epochs] # model path
test_set = False
device = '0'
for path in paths:
    model = YOLO(path)
    metrics = model.val(data="{}.yaml".format(dataset_type), imgsz=640, batch=8, conf=0.001, iou=0.5, device=device, save=True,
                        show_boxes=True, split="val")
    print ('mAP#@75 = {}'.format(metrics.box.map75))
    if test_set:
        metrics = model.val(data="{}.yaml".format(dataset_type), imgsz=640, batch=8, conf=0.001, iou=0.5, device=device, save=True,
                            show_boxes=True, split="test")
        print('mAP#@75 = {}'.format(metrics.box.map75))
    print ("*"*50)












































