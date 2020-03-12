import cv2
from pathlib import Path

from face_detector.dsfd_inference.dsfd import detect, get_face_detections


def main():
    # Initialize DSFD face detector with pre-trained weight file contained 
    # in "face_detector/dsfd_inference/dsfd/weights" directory
    weights_path = Path("face_detector/dsfd_inference/dsfd/weights/WIDERFace_DSFD_RES152.pth")
    detector = detect.DSFDDetector(weight_path=weights_path)
    print('Finished loading face detector!')

    # Loop over test images
    test_img_folder = Path("test_images/raw_images")
    faces_folder = Path("test_images/cropped_resized_faces")
    for img_path in test_img_folder.iterdir():
        # Read in image
        img = cv2.imread(str(img_path))
        print("Processing:", str(img_path))

        # Perform face detection
        # Returned bounding box data is [xmin, ymin, xmax, ymax, detection_confidence]
        detections = detector.detect_face(img, confidence_threshold=0.5)

        # # This function allows setting additional parameters for increasing
        # # robustness in face detection, but at huge inference speed costs
        # detections = get_face_detections(
        #     detector,
        #     img,
        #     confidence_threshold=0.5,
        #     use_multiscale_detect=False,
        #     use_image_pyramid_detect=False,
        #     use_flip_detect=False)
        
        # Crop face images based on bounding boxes
        for face_idx in range(len(detections)):
            x_min = detections[face_idx][0].astype(int)
            y_min = detections[face_idx][1].astype(int)
            x_max = detections[face_idx][2].astype(int)
            y_max = detections[face_idx][3].astype(int)

            face = img[y_min:y_max, x_min:x_max]

            # Resize face to 160x160 px for best results with
            # face classifier since it was trained on 160x160 px faces
            face = cv2.resize(face, (160, 160))

            # Save face image
            output_name = faces_folder.joinpath(img_path.stem + '_{}'.format(face_idx) + img_path.suffix)
            cv2.imwrite(str(output_name), face)
    
    print("Finished!")


if __name__ == "__main__":
    main()
