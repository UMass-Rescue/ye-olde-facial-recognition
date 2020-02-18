from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from matplotlib import pyplot as plt
import shutil
from pathlib import Path

def main():
    # Load previously generated embeddings
    print("Loading encodings...")
    data = pickle.loads(open("encodings.pickle", "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    encodings = np.asarray(encodings).squeeze()

    # neigh = NearestNeighbors(n_neighbors=3, metric='cosine')
    # nbrs = neigh.fit(encodings)
    # distances, indices = nbrs.kneighbors(encodings)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:,2]
    # # plt.plot(distances)
    # # plt.show()

    # # Cluster the embeddings with DBSCAN
    # print("Clustering...")
    # clt = DBSCAN(eps=0.15, metric="euclidean", min_samples=3)
    # clt.fit(encodings)
    # label_ids = np.unique(clt.labels_)

    kmeans = KMeans(n_clusters=9, random_state=0).fit(encodings)
    label_ids = np.unique(kmeans.labels_)
    labels = kmeans.labels_
    # Determine the total number of unique faces found in the dataset
    # label_ids = np.unique(clt.labels_)
    # num_unique_faces = len(np.where(label_ids > -1)[0])
    # print("Number of unique faces: {}".format(num_unique_faces))

    # Split images into clusters based on labels
    image_paths = [d["image_path"] for d in data]
    output_folder = Path("test_images/clustered_images")
    for i in range(len(image_paths)):
        current_label = labels[i]
        current_file = image_paths[i]
        new_path = output_folder.joinpath(str(current_label) + "_" + current_file.name)
        shutil.copy(current_file, new_path)

    print("Finished!")

if __name__ == "__main__":
    main()