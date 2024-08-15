import argparse
import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.plots import colors, Annotator

def check_img_size(imgsz, s=32, min_s=16):
    if isinstance(imgsz, int):
        imgsz = max(round(imgsz / s) * s, min_s)
    elif isinstance(imgsz, tuple) and len(imgsz) == 2:
        imgsz = (max(round(imgsz[0] / s) * s, min_s), max(round(imgsz[1] / s) * s, min_s))
    return imgsz

def yolov5_detect(im0, device, model, conf_thres=0.25, iou_thres=0.45, max_det=1000,
                  classes=None, agnostic_nms=False, imgsz=(640, 640), pt=True):
    imgsz = check_img_size(imgsz, s=stride)
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    
    if im is None:
        return None, im0  # Handle the case where im is None
    
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255
    
    if len(im.shape) == 3:
        im = im[None]

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    if pred[0] is None:
        return None, im

    return pred, im

def predict_realtime():
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None)
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture('http://10.140.120.189:8080/video')  # Use the IP Webcam stream
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        pred, img = yolov5_detect(frame, device='cpu', model=model,
                                  conf_thres=conf_thres,
                                  iou_thres=iou_thres)

        for i, det in enumerate(pred):
            s = ''
            annotator = Annotator(frame, line_width=3, example=str(names))
            labels = []

            if det is None:
                break

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
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

    weights = args.weights_path
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'
    imgsz = args.frame_size

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    
    if half:
        model.half()
    
    conf_thres = 0.5
    iou_thres = 0.5

    if args.option == 'realtime':
        predict_realtime()
