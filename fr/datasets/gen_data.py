import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from sklearn import cluster, datasets, mixture
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def gen_data(n=1000, is_show = False, data_type ='s-curve1',  random_state=42):
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

	if data_type == 's-curve':
		S_points, S_color = datasets.make_s_curve(n_samples, random_state=random_state)
		X, y= S_points, S_color
		plot_3d(S_points, S_color, title='manifold 3d')
	elif data_type == 'mnist':
		in_dir = 'datasets/MNIST/mnist'
		file = os.path.join(in_dir, 'mnist_train.csv')
		df = pd.read_csv(file)
		X_train = df.iloc[:, 1:].values
		y_train = df.label.values

		# file = os.path.join(in_dir, 'mnist_test.csv')
		# df = pd.read_csv(file)
		# X_test = df.iloc[:, 1:].values
		# y_test = df.label.values
		X_train_, X_, y_train_, y_ = train_test_split(X_train, y_train,
		                                                    train_size=n_samples * 10, shuffle=True,
		                                                    stratify = y_train,
			                                            random_state=random_state)  # train set = 1-ratio
		X, y = X_train_, y_train_
	else:
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

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


if __name__ == '__main__':
	gen_data()