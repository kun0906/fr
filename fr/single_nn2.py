"""
	python3 single_nn2.py

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import bisect
import collections
import copy
import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from torch.optim.lr_scheduler import ExponentialLR

from datasets import gen_data

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

np.set_printoptions(precision=3)

print(datetime.datetime.now())

class Net(nn.Module):

	def __init__(self, in_dim, out_dim=2):
		super(Net, self).__init__()
		# an affine operation: y = Wx + b
		dim = max(in_dim, out_dim * 8 * 2)
		self.fc1 = nn.Linear(in_dim, dim // 4)
		self.fc2 = nn.Linear(dim // 4, dim // 8)
		self.fc20 = nn.Linear(dim // 8, dim // 8)
		self.fc3 = nn.Linear(dim // 8, out_dim)

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		# x = F.leaky_relu(self.fc20(x))
		# x = F.leaky_relu(self.fc20(x))
		# x = F.leaky_relu(self.fc20(x))
		x = F.leaky_relu(self.fc20(x))
		# x = F.sigmoid(self.fc3(x))
		# x = F.softmax(self.fc3(x))
		x = self.fc3(x)
		return x


def dist2_tensor(X1, X2):
	return (X1 - X2).pow(2).sum()


def cos_sim_torch(x1, x2):
	a = torch.norm(x1)
	b = torch.norm(x2)
	if a == 0:
		x1 = x1 + 1e-5
		a = torch.norm(x1)
	elif b == 0:
		x2 = x2 + 1e-5
		b = torch.norm(x2)

	return torch.dot(x1, x2) / (a * b)


def t_distribute_torch(x1, x2):
	return 1 / (1 + (x1 - x2).pow(2).sum())


def gaussian_torch(x1, x2, sigma=1):
	return torch.exp(-(x1 - x2).pow(2).sum() / (2 * sigma ** 2))


def compute_j_i(X, i, sigma_i=1):
	# for each xi
	loss = 0
	N, d = X.shape
	sum_x_ji_ = 0  # s_j|i: xi will pick xj as its neighbor, under a Gaussian centered at xi
	for k in range(N):
		if i == k: continue
		sum_x_ji_ += gaussian_torch(X[i], X[k], sigma_i).detach().numpy()

	return sum_x_ji_


def compute_i_j(X, j, sigma_j=1):
	# for each xj
	N, d = X.shape
	sum_x_ij_ = 0  # s_i|j: xj will pick xi as its neighbor, under a Gaussian centered at xj
	for k in range(N):
		if j == k: continue
		sum_x_ij_ += gaussian_torch(X[j], X[k], sigma_j).detach().numpy()

	return sum_x_ij_


def compute_y_kl(X, net):
	N, d = X.shape
	sum_y_kl = 0
	for k in range(N):
		y_k = net(X[k])
		for l in range(N):
			if l == k: continue
			y_l = net(X[l])
			sum_y_kl += t_distribute_torch(y_k, y_l).detach().numpy()
	return sum_y_kl


def plot_xy(X, y, net, epoch):
	# X = torch.from_numpy(X).float()
	O = net(X)
	O = O.detach().numpy()
	plt.figure(figsize=(5, 4))
	plt.scatter(O[:, 0], O[:, 1], c=['g' if v == 0 else 'r' for v in y])
	for i in range(O.shape[0]):
		txt = f'{i}:{X[i].detach().numpy()}->{O[i]}'
		plt.annotate(txt, (O[i, 0], O[i, 1]))
	plt.title(f'epoch: {epoch}')
	plt.tight_layout()
	# plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close()

def compute_entropy(X, i, sigma):
	N, d = X.shape
	# p_ji_ = gaussian_torch(X[i], X[j], sigma)/compute_j_i(X, i, sigma_i=sigma)
	# H (P_i) = torch.sum([- p_ji_ * np.log2(p_ji_) for j in range(N) ])
	H = 0
	for j in range(N):
		if i == j: continue
		p_ji_ = gaussian_torch(X[i], X[j], sigma)/compute_j_i(X, i, sigma_i=sigma)
		if p_ji_ == 0:
			H += 0
		else:
			H += -1 * p_ji_ * torch.log2(p_ji_)

	return H


def compute_sigma_i(X, i, perplexity=5, verbose=1):
	"""

	:param p_ji_:
		p_j|i: Xi chooses Xj as a neighbour, under a Gaussian centered at Xi
	:param perplexity:
		perplexity = 2 ** (H(P_i))
		where, P_i = - sum (p_j|i * log2(p_j|i))   over all j.

	:param verbose:
	:return:
	"""

	l = 0
	r = 100
	cnt = 0
	# the perplexity increases monotonically with the variance σi (sigma**2), so we can use binary search.
	while l < r:
		m = r - (r-l)/2  # choose m as sigma and check if meet the given perplexity
		H_p_i = compute_entropy(X, i, sigma=m)   # H(P_i)
		if 2**H_p_i >= perplexity:
			r = m
		else:
			l = m+1
		cnt +=1
	sigma_i = l
	if verbose >= 5:
		print(f'i: {i}, where, l:{l} =? r:{r}, it takes cnt ({cnt}) steps to converge. Sigma_i: {l}')

	# # from sklearn: not dig into it yet.
	# # Compute conditional probabilities such that they approximately match
	# # the desired perplexity
	# distances.sort_indices()
	# n_samples = distances.shape[0]
	# distances_data = distances.data.reshape(n_samples, -1)
	# distances_data = distances_data.astype(np.float32, copy=False)
	# conditional_P = sklearn.manifold._utils._binary_search_perplexity(
	# 	distances_data, desired_perplexity, verbose
	# )
	# assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

	return sigma_i


def compute_sigma_j(X, j, perplexity=5, verbose=1):
	"""

	:param p_ij_:
		p_i|j: Xj chooses Xi as a neighbour, under a Gaussian centered at Xj
	:param perplexity:
		perplexity = 2 ** (H(P_j))
		where, P_j = - sum (p_i|j * log2(p_i|j))   over all i.

	:param verbose:
	:return:
	"""

	l = 0
	r = 100
	cnt = 0
	# the perplexity increases monotonically with the variance σi (sigma**2), so we can use binary search.
	while l < r:
		m = r - (r-l)/2  # choose m as sigma and check if meet the given perplexity
		H_p_j = compute_entropy(X, j, sigma=m)   # H(P_i)
		if 2**H_p_j >= perplexity:
			r = m
		else:
			l = m+1
		cnt +=1
	sigma_j = l
	if verbose >= 5:
		print(f'j: {j}, where, l:{l} =? r:{r}, it takes cnt ({cnt}) steps to converge. Sigma_j: {l}')

	return sigma_j

def main():
	out_dir = 'out'
	data_name = '5gaussians-5dims'
	X_raw, y_raw = gen_data.gen_data(n=1, data_type= data_name, is_show=False, random_state=42)
	# X_raw = np.asarray([[0,0], [1, 3], [2, 0], [-3, 3]])
	# y_raw = np.asarray([0, 0, 1, 1])
	print(X_raw.shape, collections.Counter(y_raw))

	# X, y = X_raw, y_raw
	# tsne = TSNE(random_state=42)
	# X_ = tsne.fit_transform(X)
	#
	# plt.scatter(X_[:, 0], X_[:, 1], c=y)
	# plt.title('TSNE X')
	# plt.show()
	#
	# umap = UMAP(random_state=42)
	# X_ = umap.fit_transform(X)
	#
	# plt.scatter(X_[:, 0], X_[:, 1], c=y)
	# plt.title('UMAP X')
	# plt.show()
	#
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

	# for r in range(10):
	# 	indices = (y_raw == r)
	# 	x = X_raw[indices]
	# 	d = pdist(x)
	# 	# print(collections.Counter(d).items())
	# 	print(np.quantile(d, q=[0, 0.3, 0.5, 0.7,1.0]))
	# 	for c in range(r+1, 10):
	# 		x2 = X_raw[y_raw==c]
	# 		d = cdist(x, x2)
	# 		print(r, c, np.quantile(d, q=[0, 0.3, 0.5, 0.7, 1.0]))

	n, d = X_raw.shape
	out_dim = 2
	n_epochs = 200
	model_file = os.path.join(out_dir, f'net_ep_{n_epochs}-single-nn.pt')
	if os.path.exists(model_file):
		os.remove(model_file)

	ds = pdist(X_raw)
	d_u = np.max(ds)
	print(f'd_u: {d_u}')
	if not os.path.exists(model_file):
		net = Net(in_dim=d, out_dim=out_dim)
		print(net)
		params = list(net.parameters())
		print(len(params))
		print(params[0].size())  # conv1's .weight
		# create your optimizer
		optimizer = optim.Adam(net.parameters(), lr=0.05)
		# optimizer = optim.SGD(net.parameters(), lr=0.01)
		criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		batch_size = 32
		perplexity = 5
		history = {'loss': []}
		for epoch in range(n_epochs + 1):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			N, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
			X = torch.from_numpy(X).float()
			L = 0

			optimizer.zero_grad()  # zero the gradient buffers
			loss = 0
			sum_y_ij = compute_y_kl(X, net)
			seen = {}
			ls = []
			sigmas = {}
			for i in range(N):
				if i not in sigmas:   #save time
					# Using binary search to find sigma_i that meets the desired perplexity
					sigma_i = compute_sigma_i(X, i, perplexity)
					sigmas[i] = sigma_i
				else:
					sigma_i = sigmas[i]
				# sigma_i = compute_sigma_i(X, i, perplexity, verbose=1)
				sum_x_ji_ = compute_j_i(X, i, sigma_i)  # for fixed i
				sum_p_ji = 0

				y_i = net(X[i])
				for j in range(0, N):
					if i == j: continue
					if j not in sigmas: # save time
						sigma_j = compute_sigma_j(X, j, perplexity)
						sigmas[j] = sigma_j
					else:
						sigma_j = sigmas[j]
					# sigma_j = compute_sigma_j(X, j, perplexity, verbose=1)
					sum_x_ij_ = compute_i_j(X, j, sigma_j)  # for fixed j
					p_ji_ = gaussian_torch(X[i], X[j], sigma_i) / sum_x_ji_  # p_j|i
					p_ij_ = gaussian_torch(X[j], X[i], sigma_j) / sum_x_ij_  # p_i|j
					p_ij = (p_ji_ + p_ij_) / (2 * N)
					# print(p_ji_, p_ij_, p_ij)
					sum_p_ji += p_ji_

					y_j = net(X[j])
					q_ij = t_distribute_torch(y_i, y_j) / sum_y_ij

					loss += torch.clamp((p_ij * torch.log(p_ij / q_ij + 1e-10).abs()), min=-1e-20, max=1e+20)
					# print(loss, p_ij, q_ij, torch.log(p_ij / q_ij))
			# loss += p_ij * torch.log(p_ij / q_ij).abs() \
			#         + dist2_tensor(dist2_tensor(X[i], X[j]), dist2_tensor(y_i, y_j)) \
			#         + dist2_tensor(cos_sim_torch(X[i], X[j]), cos_sim_torch(y_i, y_j))

			# print(i, sum_p_ji)

			# plot_xy(X, y, net, epoch
			loss.backward()
			optimizer.step()  # Does the update
			L += loss.item()
			print(f'{epoch + 1}/{n_epochs}, loss: {L}, sigma_i: {sigma_i}')
			history['loss'].append(L)
			if epoch % 50 == 0:
				print(f'*** lr:{scheduler.get_lr()}')
				scheduler.step()  # adjust learning rate
		# plot_xy(X, y, net, epoch)

		f = os.path.join(out_dir, f'{data_name}-loss.png')
		plt.plot(history['loss'])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

		# save model
		with open(model_file, 'wb') as f:
			torch.save(net, f)
	else:
		net = torch.load(model_file)

	for i in range(N):
		for j in range(i + 1, N):
			y_i = net(X[i])
			y_j = net(X[j])
			print(i, j, dist2_tensor(X[i], X[j]), dist2_tensor(y_i, y_j))

	f = os.path.join(out_dir, f'{data_name}-KL.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5))  # width, height

	m = N // N  # N//5
	indices = range(0, N, m)
	X, y = X_raw[indices, :], y_raw[indices]
	if len(set(y)) != 2:
		ax[0].scatter(X[:, 0], X[:, 1], c = y)
	else:
		ax[0].scatter(X[:, 0], X[:, 1], c=['g' if v == 0 else 'r' for v in y])
	# for i in range(X.shape[0]):
	# 	txt = f'{i}:{X[i]}'
	# 	ax[0].annotate(txt, (X[i, 0], X[i, 1]))
	ax[0].set_title(f'Original Space')

	X, y = X_raw, y_raw
	X = torch.from_numpy(X).float()
	O = net(X)
	O = O.detach().numpy()
	if len(set(y)) != 2:
		ax[1].scatter(O[:, 0], O[:, 1], c=y)
	else:
		ax[1].scatter(O[:, 0], O[:, 1], c=['g' if v == 0 else 'r' for v in y])
	# for i in range(O.shape[0]):
	# 	txt = f'{i}:{X[i].detach().numpy()}->{O[i]}'
	# 	ax[1].annotate(txt, (O[i, 0], O[i, 1]))
	ax[1].set_title('Lower space')

	fig.suptitle('KL')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close()


if __name__ == '__main__':
	main()
