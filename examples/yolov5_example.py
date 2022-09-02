import cv2

from yolov5.detector import YoloV5Detector
from yolov5.utils.plots import plot_one_box


def image_example():
    yolo = YoloV5Detector()
    images = ["inference/images/bus.jpg", "inference/images/horses.jpg", "inference/images/image1.jpg",
              "inference/images/image2.jpg"]
    for image in images:
        img = cv2.imread(image)
        img = detect_and_plot(yolo, img)
        cv2.imshow("Image", img)
        cv2.waitKey(0)


def detect_and_plot(model, image):
    targets = ['car', 'motorcycle', 'truck', 'bus']
    # image = cv2.resize(image, (1920, int(image.shape[0]/(image.shape[1] / 1920))))
    print(image.shape)
    detections = model.detect(image, class_labels=targets)
    for detection in detections:
        label = f'{detection.label} {detection.confidence:.2f}'
        plot_one_box(detection.bbox, image, label=label, color=model.colors[detection.class_id], line_thickness=3)
    return image


def video_example():
    import time
    yolo = YoloV5Detector()
    url = "/home/ashok/Documents/identeq_data/vehicle_videos/VID_20220729_085033.mp4"
    url = "/home/ashok/Documents/identeq_data/vehicle_videos/back_camera/recordings/20220901163847623_DS-2CD2T63G0-I5 treeleaf_C60765827.mp4"
    url = "/home/ashok/Documents/identeq_data/vehicle_videos/back_camera/recordings/20220901170309805_DS-2CD2T63G0-I5 treeleaf_C60765827.mp4"
    cap = cv2.VideoCapture(url)
    assert cap.isOpened(), f'Failed to open {url}'
    fps = cap.get(cv2.CAP_PROP_FPS) % 100
    print("FPS:", fps)
    start_time = time.time()
    processed_frames = 0
    skip_num_frames = 3
    while cap.isOpened():
        if skip_num_frames > 0:
            for i in range(skip_num_frames):
                cap.grab()
        else:
            cap.grab()
        success, img = cap.retrieve()
        if not success:
            break
        t = time.time()
        img = detect_and_plot(yolo, img)
        processed_frames += 1
        t = time.time() - t
        end_time = time.time()
        processing_fps = processed_frames / (end_time - start_time)
        cv2.imshow("Image", img)
        time.sleep(1 / fps)
        print(f"Time: {t:.3f} sec.")
        print(f"Processing FPS: {processing_fps:.2f}")
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_example()
