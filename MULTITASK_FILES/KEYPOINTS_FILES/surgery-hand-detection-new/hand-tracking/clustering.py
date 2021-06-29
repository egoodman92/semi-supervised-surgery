import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# uses GT data


def generate_array(keypoints):
	arr = np.zeros((210, ))
	idx = 0
	for i in range(21):
		for j in range(i + 1, 21):
			if keypoints[i * 3 + 2] > 0 and keypoints[j * 3 + 2] > 0:
				arr[idx] = np.sqrt(((keypoints[i * 3] - keypoints[j * 3]) ** 2) + ((keypoints[i * 3 + 1] - keypoints[j * 3 + 1]) ** 2))
				idx += 1

	return arr


def plot_data():
	data = json.load(open("../keypoint-analysis/constants/coco_validated_full_8_20.json", "r"))['annotations']


	arrays = []
	for ann in data:
		arrays.append(generate_array(ann['keypoints']) / ann['area'])

	# PLOT PCA in 2D
	k = KMeans(n_clusters=5)
	k.fit(np.vstack(arrays))

	pca = PCA(n_components=2)
	reduced = pca.fit_transform(np.vstack(arrays))

	t = reduced.transpose()

	plt.scatter(t[0], t[1], c=k.labels_.astype(np.float))
	plt.show()





plot_data()