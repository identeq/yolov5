import cv2

from yolov5.detector import YoloV5Detector
from yolov5.utils.plots import plot_one_box


def image_example():
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


def detect_and_plot(model, image):
    boxes, class_ids, confidences = model.detect(image)
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        label = f'{model.labels[cls_id]} {conf:.2f}'
        plot_one_box(box, image, label=label, color=model.colors[cls_id], line_thickness=3)
    return image


def video_example():
    import time
    yolo = YoloV5Detector()
    url = "/home/ashok/Documents/identeq_data/vehicle_videos/VID_20220729_085033.mp4"
    cap = cv2.VideoCapture(url)
    assert cap.isOpened(), f'Failed to open {url}'
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) % 100
    print("FPS:", fps)
    start_time = time.time()
    end_time = time.time()
    processing_fps = 0
    processed_frames = 0
    n = 0
    while cap.isOpened():
        n += 1
        cap.grab()
        if n % 4 == 0:  # read every 4th frame
            success, img = cap.retrieve()
            if not success:
                break
            t = time.time()
            img = detect_and_plot(yolo, img)
            processed_frames += 1
            t = time.time() - t
            print(f"Time: {t:.3f} sec.")
            cv2.imshow("image", img)
            cv2.waitKey(1)
            n = 0
        time.sleep(1 / fps)  # wait time
        end_time = time.time()
        processing_fps = processed_frames / (end_time - start_time)
        print(f"Processing FPS: {processing_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_example()
