import argparse

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.datasets import LoadImages
from utils.general import (Profile, check_img_size, non_max_suppression, scale_coords)
from utils.general import set_logging
from utils.plots import colors, Annotator
from utils.torch_utils import select_device


# Instantiate the parser


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


def predict_image(img_path, path_save):
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride)  # load image to yolo processing
    img_result, labels = load_model_yolo(model, device, dataset)  # feed image to model and get output
    img_result = cv2.resize(img_result, (640, 640))  # resize image to save
    cv2.imwrite(path_save, img_result)


def yolov5_detect(im0, device, model, conf_thres=0.25, iou_thres=0.45, max_det=1000,
                  classes=None, agnostic_nms=False, imgsz=(640, 640), pt=True):
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    with dt[1]:
        pred = model(im, augment=False, visualize=False)
        # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return pred, im


def predict_video(video_path, path_save):
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_save, fourcc, 20.0, (640, 480))
    cap = cv2.VideoCapture(video_path)
    names = model.module.names if hasattr(model, 'module') else model.names
    while (cap.isOpened()):
        ret, frame = cap.read()
        pred, img = yolov5_detect(frame, device='cpu', model=model,
                                  conf_thres=conf_thres,
                                  iou_thres=iou_thres)

        for i, det in enumerate(pred):  # detections per image
            s = ''
            annotator = Annotator(frame, line_width=3, example=str(names))
            labels = []
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
        cv2.imshow('prediction', added_image)
        out.write(added_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--option', help='image or video', default='image', type=str)
    parser.add_argument('--weights_path', help='weight path', default='weights/best.pt', type=str)
    parser.add_argument('--saved_path', help='result path', type=str)
    parser.add_argument('--image_path', help='image detection', default='test.jpg', type=str)
    parser.add_argument('--video_path', help='video detection', default='test.avi', type=str)
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
    if args.option == 'image':
        if args.saved_path == '':
            saved_path = "result/result.jpg"
        else:
            saved_path = args.saved_path
        predict_image(args.image_path, saved_path)
    else:
        if args.saved_path == '':
            saved_path = "result/result.avi"
        else:
            saved_path = args.saved_path
        predict_video(args.video_path, saved_path)
