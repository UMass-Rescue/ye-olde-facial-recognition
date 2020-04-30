import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import time

from face_detector.dsfd_inference.dsfd import detect, get_face_detections


def extract_faces(
    dsfd_weights_path=None,
    input_images_path=None,
    extract_faces=True,
    save_bounding_boxes=True,
):
    # Initialize face detector
    weights_path = Path(dsfd_weights_path)
    detector = detect.DSFDDetector(weight_path=weights_path)
    print("Finished loading face detector!")

    # Input folder containing raw images
    test_img_folder = Path(input_images_path)

    # Output folder for saving cropped faces
    if extract_faces:
        faces_folder = test_img_folder.parent.joinpath("faces_" + test_img_folder.stem)
        Path(faces_folder).mkdir(parents=True, exist_ok=True)

    # Filepaths and bounding boxes
    all_bounding_boxes = []

    for img_path in test_img_folder.iterdir():
        # Read in image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Processing:", str(img_path))

        # Perform face detection
        # Returned bounding box data is [xmin, ymin, xmax, ymax, detection_confidence]
        detections = detector.detect_face(img, confidence_threshold=0.5)
        current_image_bounding_boxes = np.copy(detections)
        current_image_bounding_boxes = current_image_bounding_boxes[:, 0:4]
        current_image_bounding_boxes[:, [0, 1, 2, 3]] = current_image_bounding_boxes[
            :, [1, 2, 3, 0]
        ]
        current_image_bounding_boxes = current_image_bounding_boxes.astype(int)

        # # This function allows setting additional parameters for increasing
        # # robustness in face detection, but at huge inference speed costs
        # detections = get_face_detections(
        #     detector,
        #     img,
        #     confidence_threshold=0.5,
        #     use_multiscale_detect=False,
        #     use_image_pyramid_detect=False,
        #     use_flip_detect=False)
        # bounding_boxes = np.copy(detections)
        # bounding_boxes = bounding_boxes[:, 0:4]
        # bounding_boxes[:,[0, 1, 2, 3]] = bounding_boxes[:,[1, 2, 3, 0]].astype(int)
        # bounding_boxes = list(zip(bounding_boxes[:,0], bounding_boxes[:,1], bounding_boxes[:,2], bounding_boxes[:,3]))

        # Crop face images based on bounding boxes
        for face_idx in range(len(current_image_bounding_boxes)):
            current_face = current_image_bounding_boxes[face_idx]

            x_min = current_face[3]
            y_min = current_face[0]
            x_max = current_face[1]
            y_max = current_face[2]

            # Calculate padding
            y_dist = y_max - y_min
            x_dist = x_max - x_min
            y_padding = int(0.50 * y_dist)
            x_padding = int(0.50 * x_dist)

            # Update bounding boxes with padding
            img_shape = img.shape
            x_min = np.max([0, x_min - x_padding])
            x_max = np.min([x_max + x_padding, img_shape[1]])
            y_min = np.max([0, y_min - y_padding])
            y_max = np.min([y_max + y_padding, img_shape[0]])

            face = img[y_min:y_max, x_min:x_max]

            # Save face image
            if extract_faces:
                output_name = faces_folder.joinpath(
                    img_path.stem + "_{}".format(face_idx) + img_path.suffix
                )
                cv2.imwrite(str(output_name), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

            # Save image file path and bounding box coordinates for each
            # face detected in image as rows for later converting to dataframe
            if save_bounding_boxes:
                current_face_details = [
                    str(img_path),
                    current_face[0],
                    current_face[1],
                    current_face[2],
                    current_face[3],
                ]
                all_bounding_boxes.append(current_face_details)

    # Save image names and face bounding boxes into dataframe
    if save_bounding_boxes:
        print("Saving bounding boxes.")

        # Generate dataframe with bounding box info
        all_face_details = pd.DataFrame(
            all_bounding_boxes,
            columns=["image_path", "y_min", "x_max", "y_max", "x_min"],
        )

        # Export dataframe as CSV for processing by embedding generator
        all_face_details.to_csv(
            "all_face_bounding_boxes.csv", encoding="utf-8", index=False
        )


def main():
    # Path to DSFD weights
    dsfd_weights_path = (
        "face_detector/dsfd_inference/dsfd/weights/WIDERFace_DSFD_RES152.pth"
    )

    # Path to input images
    test_img_folder = "test_images/lotr_cast"

    # Extract faces
    start = time.time()
    extract_faces(
        dsfd_weights_path=dsfd_weights_path,
        input_images_path=test_img_folder,
        extract_faces=True,
        save_bounding_boxes=True,
    )
    end = time.time()
    print(end - start)
    print("Finished!")


if __name__ == "__main__":
    main()
