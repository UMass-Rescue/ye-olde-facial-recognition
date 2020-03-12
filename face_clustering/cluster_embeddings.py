from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from matplotlib import pyplot as plt
import shutil
from pathlib import Path
import dlib

def main():
    # Load previously generated embeddings
    print("Loading encodings...")
    data = pickle.loads(open("encodings.pickle", "rb").read())
    data = np.array(data)

    # Specifically grab the encodings from the data array
    # If using dlib's Chinese Whispers Clustering, convert to dlib vector format
    encodings = [dlib.vector(d["encoding"].squeeze()) for d in data]
    # If using KNN, keep in Numpy format
    # encodings = [d["encoding"] for d in data]
    # encodings = np.asarray(encodings).squeeze()

    # Calculate a threshold value for Chinese Whispers
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(encodings)
    distances, indices = nbrs.kneighbors(encodings)
    distances = np.sort(distances, axis=0)
    distances = distances[:,2]
    mean_distance = np.mean(distances)
    # plt.plot(distances)
    # plt.show()

    # Clustering with Chinese Whispers algorithm
    labels = dlib.chinese_whispers_clustering(encodings, mean_distance)

    # kmeans = KMeans(n_clusters=5, random_state=0).fit(encodings)
    # label_ids = np.unique(kmeans.labels_)
    # labels = kmeans.labels_

    # Determine the total number of unique faces, as well
    # as their occurrences 
    label_ids, counts = np.unique(labels, return_counts=True)
    num_unique_faces = len(label_ids)

    # Split images into clusters based on labels
    image_paths = [d["image_path"] for d in data]
    output_folder = Path("test_images/clustered_faces")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for i in range(len(image_paths)):
        current_label = labels[i]
        current_file = image_paths[i]
        new_path = output_folder.joinpath(str(current_label) + "_" + current_file.name)
        shutil.copy(current_file, new_path)

    print("Finished!")

if __name__ == "__main__":
    main()