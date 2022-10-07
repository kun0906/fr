import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from sklearn import datasets
from sklearn.model_selection import train_test_split


def gen_data(n=1000, is_show=False, data_type='s-curve1', with_noise=False, random_state=42):
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
		X, y = S_points, S_color
		plot_3d(S_points, S_color, title='manifold 3d')
	elif data_type == 'moon':
		X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
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
		                                              stratify=y_train,
		                                              random_state=random_state)  # train set = 1-ratio
		X, y = X_train_ / 255, y_train_
	elif data_type == '1gaussian':
		r = np.random.RandomState(seed=random_state)
		X = r.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_samples)
		y = np.zeros((X.shape[0],))
	elif data_type == '2gaussians':
		r = np.random.RandomState(seed=random_state)
		X1 = r.multivariate_normal([-0.5, 0], [[0.1, 0], [0, 0.1]], size=n_samples)
		y1 = np.zeros((X1.shape[0],))
		X2 = r.multivariate_normal([0.5, 0], [[0.1, 0], [0, 0.1]], size=n_samples)
		y2 = np.ones((X2.shape[0],))

		X = np.concatenate([X1, X2], axis=0)
		y = np.concatenate([y1, y2], axis=0)

	elif data_type == '5gaussians':
		r = np.random.RandomState(seed=random_state)
		for i, mu in enumerate([[-2, 0], [2, 0], [0, -2], [0, 2], [0, 0]]):
			X1 = r.multivariate_normal(mu, [[0.1, 0], [0, 0.1]], size=n_samples)
			y1 = np.ones((X1.shape[0],)) * i
			if i == 0:
				X = copy.deepcopy(X1)
				y = copy.deepcopy(y1)
			else:
				X = np.concatenate([X, X1], axis=0)
				y = np.concatenate([y, y1], axis=0)

	elif data_type == '5gaussians-5dims':
		r = np.random.RandomState(seed=random_state)
		# for i, mu in enumerate([-5, 5, -3, 3, 0]):
		# mu = np.asarray([mu] * 5)
		# cov = np.asarray([cov] * 5)
		n_clusters = 10
		for i in range(n_clusters):
			mu = r.uniform(low=0, high=5, size=n_clusters)
			cov = np.asarray([0.1] * n_clusters)
			cov = np.diag(np.array(cov))
			X1 = r.multivariate_normal(mu, cov, size=n_samples)
			y1 = np.ones((X1.shape[0],)) * i

			if i == 0:
				X = copy.deepcopy(X1)
				y = copy.deepcopy(y1)
			else:
				X = np.concatenate([X, X1], axis=0)
				y = np.concatenate([y, y1], axis=0)

	else:
		X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
		if with_noise:
			n_noise_samples = int(n_samples * 0.01)
			rng = np.random.RandomState(seed=random_state)
			X_noise = rng.multivariate_normal(mean=[2, 0], cov=np.asarray([[0.1, 0.0], [0.0, 0.1]]),
			                                  size=n_noise_samples)
			y_noise = np.asarray([2] * X_noise.shape[0])

			X = np.concatenate([X, X_noise], axis=0)
			y = np.concatenate([y, y_noise], axis=0)

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


def gen_noise(n=1000, is_show=False, data_type='s-curve1', random_state=42):
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
	rng = np.random.RandomState(seed=random_state)
	X = rng.multivariate_normal(mean=[0.5, 0], cov=np.asarray([[0.1, 0.0], [0.0, 0.1]]), size=n_samples)
	y = np.asarray(['noise'] * X.shape[0])
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
