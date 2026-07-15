import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='FRA-YOLO Training Script')
    parser.add_argument('--model', type=str, default='yolov8m-fra.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--data', type=str, default='VisDrone.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use for training (e.g., 0, cpu)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision training')
    parser.add_argument('--swanlab', action='store_true', default=True,
                        help='Enable SwanLab logging')
    parser.add_argument('--project', type=str, default='ultralytics',
                        help='SwanLab project name')
    parser.add_argument('--description', type=str, default='yolov8m on visdrone',
                        help='SwanLab experiment description')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model = YOLO(args.model)

    # Setup SwanLab logging if enabled
    if args.swanlab:
        from swanlab.integration.ultralytics import add_swanlab_callback
        add_swanlab_callback(
            model,
            project=args.project,
            experiment_name=args.model,
            description=args.description,
        )

    # Train the model
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        batch=args.batch,
        amp=args.amp,
    )
    
    # Validate the model
    metrics = model.val()


if __name__ == '__main__':
    main()



