import cv2
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import time

import face_recognition


def generate_embeddings(bounding_boxes_path=None, input_images_path=None):
    input_images_path = Path(input_images_path)
    faces_folder = input_images_path.parent.joinpath('faces_' + input_images_path.stem)

    all_bounding_boxes = pd.read_csv(Path(bounding_boxes_path)) 

    grouped_bb = list(all_bounding_boxes.groupby([all_bounding_boxes.columns[0]]))

    output_data = []

    for i in range(len(grouped_bb)):
        current_image_data = grouped_bb[i]

        # Read in image
        current_img_path = Path(current_image_data[0])
        current_img = cv2.imread(str(current_img_path))
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)

        # Get bounding box data and convert to tuple format as needed for embedding generator
        current_bounding_boxes = np.array(current_image_data[1].drop([all_bounding_boxes.columns[0]], axis=1), dtype=np.float32)
        current_bounding_boxes = tuple(map(tuple, current_bounding_boxes))

        # Generate embeddings
        encodings = face_recognition.face_encodings(current_img, current_bounding_boxes)

        # Save embeddings into output file
        for face_idx in range(len(encodings)):
            output_name = faces_folder.joinpath(current_img_path.stem + '_{}'.format(face_idx) + current_img_path.suffix)

            # Save encoding
            d = [{"image_path": output_name, "encoding": encodings[face_idx]}]
            output_data.extend(d)
    
    # Save encodings to separate file
    print("Saving encodings.")
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(output_data))
    f.close()


def main():
    # Path to bounding box detail CSV file
    bounding_boxes_path = 'all_face_bounding_boxes.csv'

    # Path to input images
    test_img_folder = "test_images/lotr_cast"
    start = time.time()
    generate_embeddings(bounding_boxes_path=bounding_boxes_path, input_images_path=test_img_folder)
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
