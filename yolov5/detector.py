"""
-- Created by: Ashok Kumar Pant
-- Created on: 8/1/22
"""

import random
from typing import List

import torch

from yolov5.models.common import DetectMultiBackend, AutoShapeV1
from yolov5.utils.torch_utils import select_device


class YoloV5Detector:
    def __init__(self, model_name="yolov5s.pt", img_size=640, device=''):
        self.img_size = img_size
        self.device = select_device(device=device)
        self.backend_model = DetectMultiBackend(model_name, device=self.device)
        self.model = AutoShapeV1(self.backend_model).to(self.device)
        self.labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labels]
        self._id2labels = {i: label for i, label in enumerate(self.labels)}
        self._labels2ids = {label: i for i, label in enumerate(self.labels)}

    @torch.no_grad()
    def detect(self, image, thresh=0.25, iou_thres=0.45, classes=None, agnostic=False):
        results = self.model(image, size=self.img_size, conf=thresh, iou=iou_thres, classes=classes, agnostic=agnostic)
        boxes = []
        confidences = []
        class_ids = []
        for i in range(len(results.xyxy[0])):
            x0, y0, x1, y1, confidence, class_id = results.xyxy[0][i].cpu().numpy().astype(float)
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
        return boxes, class_ids, confidences

    def labels2ids(self, labels: List[str]):
        return [self._labels2ids[label] for label in labels]
