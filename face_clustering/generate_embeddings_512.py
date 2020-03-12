import torch

import cv2
from pathlib import Path
import pickle

from facenet_pytorch import InceptionResnetV1


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Inception Resnet V1 pretrained classifier
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print('Finished loading face encoder!')

    # Process each face and generate embeddings
    faces_folder = Path("test_images/cropped_resized_faces")
    
    output_data = []
    for img_path in faces_folder.iterdir():
        print("Processing:", str(img_path))
        # Read in image
        img = cv2.imread(str(img_path))

        # Convert from BGR to RGB, then to proper tensor format
        img_rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float()
        img_rgb = img_rgb.unsqueeze(0)
        img_rgb = img_rgb.permute(0, 3, 1, 2)
        img_rgb = img_rgb.to(device)

        # Calculate embedding
        embeddings = resnet(img_rgb).detach().cpu().numpy()

        # Save data
        d = [{"image_path": img_path, "encoding": embeddings}]
        output_data.extend(d)

    print("Saving encodings...")
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(output_data))
    f.close()
    
    print("Finished!")


if __name__ == "__main__":
    main()