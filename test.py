from face_detector.detector import detect_faces
from face_detector.utils import show_bboxes
from PIL import Image

def main():
    image = Image.open('test_images/21388.jpg')
    bounding_boxes, landmarks = detect_faces(image)
    image = show_bboxes(image, bounding_boxes, landmarks)
    image.show()

if __name__ == "__main__":
    main()
