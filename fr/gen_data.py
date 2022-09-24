import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def gen_data(n=1000, is_show = False, random_state=42):
	"""
	https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
	:return:
	"""

	np.random.seed(0)

	# ============
	# Generate datasets. We choose the size big enough to see the scalability
	# of the algorithms, but not too big to avoid too long running times
	# ============
	n_samples = n
	X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
	# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
	# blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
	# no_structure = np.random.rand(n_samples, 2), None

	# # Anisotropicly distributed data
	# random_state = 170
	# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
	# transformation = [[0.6, -0.6], [-0.4, 0.8]]
	# X_aniso = np.dot(X, transformation)
	# aniso = (X_aniso, y)
	#
	# # blobs with varied variances
	# varied = datasets.make_blobs(
	# 	n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
	# )

	# ============
	# Set up cluster parameters
	# ============
	plt.figure()
	# plt.subplots_adjust(
	# 	left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
	# )
	plt.scatter(X[:, 0], X[:, 1], c=y)

	if is_show: plt.show()
	plt.close()
	print('test')
	return X, y


if __name__ == '__main__':
	gen_data()
