"""
-- Created by: Ashok Kumar Pant
-- Created on: 8/1/22
"""
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import random

from yolov5.models.common import DetectMultiBackend, AutoShape
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from typing import List

import numpy as np
from numpy import random

from yolov7.utils.torch_utils import select_device


class BoundingBox:
    def __init__(self, class_id, label, confidence, bbox, image_width, image_height):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # t,l,b,r or x1,y1,x2,y2
        self.bbox_normalized = np.array(bbox) / (image_width, image_height, image_width, image_height)
        self.__x1 = bbox[0]
        self.__y1 = bbox[1]
        self.__x2 = bbox[2]
        self.__y2 = bbox[3]
        self.__u1 = self.bbox_normalized[0]
        self.__v1 = self.bbox_normalized[1]
        self.__u2 = self.bbox_normalized[2]
        self.__v2 = self.bbox_normalized[3]

    @property
    def width(self):
        return self.bbox[2] - self.__x1

    @property
    def height(self):
        return self.__y2 - self.__y1

    @property
    def center_absolute(self):
        return 0.5 * (self.__x1 + self.__x2), 0.5 * (self.__y1 + self.__y2)

    @property
    def center_normalized(self):
        return 0.5 * (self.__u1 + self.__u2), 0.5 * (self.__v1 + self.__v2)

    @property
    def size_absolute(self):
        return self.__x2 - self.__x1, self.__y2 - self.__y1

    @property
    def size_normalized(self):
        return self.__u2 - self.__u1, self.__v2 - self.__v1

    def __repr__(self) -> str:
        return f'BoundingBox(class_id: {self.class_id}, label: {self.label}, bbox: {self.bbox}, confidence: {self.confidence:.2f})'


def _postprocess(boxes, scores, classes, labels, img_w, img_h):
    if len(boxes) == 0:
        return boxes

    detected_objects = []
    for box, score, class_id, label in zip(boxes, scores, classes, labels):
        detected_objects.append(BoundingBox(class_id, label, score, box, img_w, img_h))
    return detected_objects


class YoloV5Detector:
    def __init__(self, model_name="yolov5s.pt", img_size=640, device=''):
        self.img_size = img_size
        self.device = select_device(device=device)
        self.backend_model = DetectMultiBackend(model_name, device=self.device)
        self.model = AutoShape(self.backend_model).to(self.device)
        self.labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labels]
        self._id2labels = self.labels
        self._labels2ids = {label: class_id for class_id, label in self.labels.items()}

    @smart_inference_mode()
    def detect(self, image, thresh=0.25, iou_thresh=0.45, classes=None, class_labels=None, agnostic=False):
        if not classes and class_labels:
            classes = self.labels2ids(class_labels)
        results = self.model(image, size=self.img_size, conf=thresh, iou=iou_thresh, classes=classes, agnostic=agnostic)
        boxes = []
        confidences = []
        class_ids = []
        for i in range(len(results.xyxy[0])):
            x0, y0, x1, y1, confidence, class_id = results.xyxy[0][i].cpu().numpy().astype(float)
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])
            confidences.append(confidence)
            class_ids.append(int(class_id))
        labels = [self._id2labels[class_id] for class_id in class_ids]
        return _postprocess(boxes, confidences, class_ids, labels, image.shape[1], image.shape[0])

    def labels2ids(self, labels: List[str]):
        return [self._labels2ids[label] for label in labels]
