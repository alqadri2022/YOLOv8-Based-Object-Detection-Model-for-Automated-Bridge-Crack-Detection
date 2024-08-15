import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import sys

def load_model(weights):
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    return model

def perform_inference(model, img_path, conf_thres=0.5, iou_thres=0.5):
    img0 = cv2.imread(img_path)  # BGR
    img = letterbox(img0, new_shape=640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img =  img / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    if len(pred) > 0:
        pred = pred[0]
        pred = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])

    cv2.imshow('Inference Result', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <D:/documents/zhonglujiaoke/datasets/YOLO/修复后的裂缝提取/weights/best.pt> <D:/documents/zhonglujiaoke/datasets/YOLO/修复后的裂缝提取/data/val/13531139_3.jp>")
        sys.exit(1)

    weights_path = sys.argv[1]
    img_path = sys.argv[2]

    model = load_model(weights_path)
    perform_inference(model, img_path)
