import cv2
from pathlib import Path
import pickle
import numpy as np

import face_recognition
from face_detector.dsfd_inference.dsfd import detect, get_face_detections

# Initialize face detector
weights_path = Path("face_detector/dsfd_inference/dsfd/weights/WIDERFace_DSFD_RES152.pth")
detector = detect.DSFDDetector(weight_path=weights_path)
print('Finished loading face detector!')

# Input folder containing raw images
test_img_folder = Path("test_images/raw_images_1")

# Output folder for saving cropped faces
faces_folder = test_img_folder.parent.joinpath('cropped_faces_' + test_img_folder.stem)
Path(faces_folder).mkdir(parents=True, exist_ok=True)

# Process all raw images in provided folder
output_data = []
for img_path in test_img_folder.iterdir():
    # Read in image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Processing:", str(img_path))
    
    # Detect faces and convert to 128-dimensional encodings
    detections = detector.detect_face(img, confidence_threshold=0.5)
    bounding_boxes = np.copy(detections)
    bounding_boxes = bounding_boxes[:, 0:4]
    bounding_boxes[:,[0, 1, 2, 3]] = bounding_boxes[:,[1, 2, 3, 0]].astype(int)
    bounding_boxes = list(zip(bounding_boxes[:,0], bounding_boxes[:,1], bounding_boxes[:,2], bounding_boxes[:,3]))
    encodings = face_recognition.face_encodings(img, bounding_boxes)

    for face_idx in range(len(encodings)):
        x_min = detections[face_idx][0].astype(int)
        y_min = detections[face_idx][1].astype(int)
        x_max = detections[face_idx][2].astype(int)
        y_max = detections[face_idx][3].astype(int)

        face = img[y_min:y_max, x_min:x_max]

        # Resize face to 160x160 px for best results with
        # face classifier since it was trained on 160x160 px faces
        #face = cv2.resize(face, (160, 160))

        # Save cropped face image
        output_name = faces_folder.joinpath(img_path.stem + '_{}'.format(face_idx) + img_path.suffix)
        cv2.imwrite(str(output_name), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        # Save encoding
        d = [{"image_path": output_name, "encoding": encodings[face_idx]}]
        output_data.extend(d)

# Save encodings to separate file
print("Saving encodings...")
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(output_data))
f.close()

print("Finished!")

