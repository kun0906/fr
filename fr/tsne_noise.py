"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import sklearn.utils
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from datasets import gen_data
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)

def main():
	out_dir = 'out'
	# data_name = '2gaussians'
	data_name = '2circles'
	# data_name = 's-curve'
	# data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	data_name = '3gaussians-10dims'
	X, y = gen_data.gen_data(n=3, data_type= data_name, is_show=True, with_noise=True, random_state=42)
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	for seed in range(3):
		n, d = X.shape
		X, y = sklearn.utils.shuffle(X, y, random_state=seed)
		tsne = TSNE(perplexity=min(30, n-1), method='exact', random_state=42)
		X_ = tsne.fit_transform(X)

		# plt.scatter(X_[:, 0], X_[:, 1])
		plt.scatter(X_[:, 0], X_[:, 1], c=y)
		for i in range(n):
			txt = f'{i}:{X_[i]}'
			plt.gca().annotate(txt, (X_[i, 0], X_[i, 1]))
		plt.title(f'TSNE X: seed={seed}')
		plt.show()

		# umap = UMAP(random_state=42)
		# X_ = umap.fit_transform(X)
		#
		# plt.scatter(X_[:, 0], X_[:, 1], c=y)
		# plt.title('UMAP X')
		# plt.show()

		# pca = PCA(n_components=0.99, random_state=42)
		# pca.fit(X)
		# print(pca.explained_variance_, pca.explained_variance_ratio_)
		#
		# pca = PCA(n_components=2, random_state=42)
		# X_ = pca.fit_transform(X)
		#
		# plt.scatter(X_[:, 0], X_[:, 1], c=y)
		# plt.title('PCA X')
		# plt.show()



if __name__ == '__main__':
	main()
