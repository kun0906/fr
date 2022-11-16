"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
# Author: kun88.yang@gmail.com

import random

import numpy as np
import torch

import tsne_trustworthiness
from datasets import gen_data

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
	X, y = gen_data.gen_data(n=5, data_type=data_name, is_show=True, with_noise=True, random_state=42)
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	tsne = tsne_trustworthiness.TSNE(perplexity=3, method='trustworthiness', random_state=42)
	X_ = tsne.fit_transform(X, y)

	plt.scatter(X_[:, 0], X_[:, 1], c=y)
	plt.title('TSNE X')
	plt.show()


if __name__ == '__main__':
	main()
