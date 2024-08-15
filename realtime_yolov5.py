import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.datasets import LoadStreams
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.plots import colors, Annotator

def check_img_size(imgsz, s=32, min_s=16):
    if isinstance(imgsz, int):
        # Ensure imgsz is a multiple of stride
        imgsz = max(round(imgsz / s) * s, min_s)
    elif isinstance(imgsz, tuple) and len(imgsz) == 2:
        # Ensure both dimensions are multiples of stride
        imgsz = (max(round(imgsz[0] / s) * s, min_s), max(round(imgsz[1] / s) * s, min_s))
    return imgsz


def load_model_yolo(model, device, dataset):
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    print(dataset)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            annotator = Annotator(im0, line_width=3, example=str(names))
            labels = []
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    labels.append(label)
                    annotator.box_label(xyxy, label, color=colors(c, True))
    return annotator.result(), labels

def yolov5_detect(im0, device, model, conf_thres=0.25, iou_thres=0.45, max_det=1000,
                  classes=None, agnostic_nms=False, imgsz=(640, 640), pt=True):
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    seen, windows = 0, []
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Check if there are predictions
    if pred[0] is None:
        return None, im

    return pred, im

def predict_realtime():
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None)
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam
    while True:
        ret, frame = cap.read()
        pred, img = yolov5_detect(frame, device='cpu', model=model,
                                  conf_thres=conf_thres,
                                  iou_thres=iou_thres)

        for i, det in enumerate(pred):  # detections per image
            s = ''
            annotator = Annotator(frame, line_width=3, example=str(names))
            labels = []
            
            # No predictions
            if det is None:
                break

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    labels.append(label)
                    annotator.box_label(xyxy, label, color=colors(c, True))
            added_image = annotator.result()
        cv2.imshow('Real-time Object Detection', added_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--option', help='image or video', default='image', type=str)
    parser.add_argument('--weights_path', help='weight path', default='weights/best.pt', type=str)
    parser.add_argument('--saved_path', help='result path', type=str)
    parser.add_argument('--frame_size', help='size', default=704, type=int)

    args = parser.parse_args()

    # Load model
    weights = args.weights_path
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'
    imgsz = args.frame_size

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    conf_thres = 0.5
    iou_thres = 0.5

    if args.option == 'realtime':
        predict_realtime()
