# based on https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/blob/main/yolov7/utils.py
import time

import cv2
import numpy as np
import onnxruntime

class_names = ['armor', 'base', 'car', 'rune', 'rune-blue', 'rune-gray', 'rune-grey', 'rune-red', 'watcher']
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


class YOLOv8:

    def __init__(self, path, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # Initialize model
        self.class_ids = None
        self.scores = None
        self.boxes = None
        self.img_height = None
        self.img_width = None
        self.session = None
        self.initialize_model(path, providers)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path, providers):
        self.session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(path, providers=providers)

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes = xywh2xyxy(self.rescale_boxes(np.squeeze(outputs[0])))
        self.scores = np.squeeze(outputs[1])
        self.class_ids = np.squeeze(outputs[2])
        print(self.boxes)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        if self.class_ids.size == 1:
            return image
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('invalid number of arguments, expected 2:\npython3 YOLOv8.py [model_path] [image_path|video_idx]')
        exit(1)

    model_path = sys.argv[1]

    detector = YOLOv8(model_path)

    try:
        img = cv2.imread(sys.argv[2])
        if img is None:
            print("Image could not be loaded")
            exit(2)
        boxes, scores, ids = detector(img)
        print('scores:', scores)
        combined_img = detector.draw_detections(img)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", combined_img)
        cv2.waitKey(0)
    except cv2.error:
        camera = cv2.VideoCapture(int(sys.argv[2]))
        while True:
            ret, img = camera.read()
            boxes, scores, ids = detector(img)
            combined_img = detector.draw_detections(img)
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Output", combined_img)
            if cv2.waitKey(0):
                break
