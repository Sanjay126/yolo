

from yolo.yolo import YOLO, detect_img


def _main(img):
    boxes=detect_img(YOLO(),img)
    return boxes


