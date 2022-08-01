import cv2

from yolov5.detector import YoloV5Detector

if __name__ == '__main__':
    yolo = YoloV5Detector()
    images = ["inference/images/bus.jpg", "inference/images/horses.jpg", "inference/images/image1.jpg",
              "inference/images/image2.jpg"]
    for image in images:
        img = cv2.imread(image)
        boxes, class_ids, confidences = yolo.detect(img)
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = str(yolo.labels[cls_id]) + ": {:.2f}".format(conf)
            cv2.putText(img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
