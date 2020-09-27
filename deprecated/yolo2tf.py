from utils import save_tf

if __name__ == "__main__":

    config = {
        'weights': 'data/models/YOLO/yolov4_imbalanced.weights',
        'input_size': 416,
        'score_thres': 0.1,
        'model': 'yolov4',
        'weights_tf': 'data/models/YOLO/checkpoints/yolov4_imbalanced',
        'output_path': 'data/result.jpg',
        'iou': 0.45
    }

    save_tf(config)