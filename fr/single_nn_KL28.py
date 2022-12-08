""" A neural network-based dimensionality reduction method


"""
# Author: Kun.bj@outlook.com

import collections
import copy
import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import ExponentialLR
from umap import UMAP

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
		dim = max(in_dim, out_dim * 8 * 10)
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
		# x = F.leaky_relu(self.fc20(x))
		# x = F.sigmoid(self.fc3(x))
		# x = F.softmax(self.fc3(x))
		x = self.fc3(x)
		return x

		# x = torch.tanh(self.fc1(x))
		# x = torch.tanh(self.fc2(x))
		# # x = F.leaky_relu(self.fc20(x))
		# # x = F.leaky_relu(self.fc20(x))
		# # x = F.leaky_relu(self.fc20(x))
		# x = torch.tanh(self.fc20(x))
		# # x = F.sigmoid(self.fc3(x))
		# # x = F.softmax(self.fc3(x))
		# x = self.fc3(x)
		# return x


def dist2_tensor(X1, X2):
	# return (X1 - X2).pow(2).sum()
	# return (X1 - X2).abs().sum()
	k = 1
	return (X1 - X2).abs().pow(k).sum().pow(1/k)


def dist2(X1, X2):
	# return np.sum(np.square(X1 - X2))
	k = 1
	return np.sum(np.abs(X1 - X2)**(k)) ** (1/k)

# sigma = 1
# return torch.exp(-(X1 - X2).pow(2).sum() / (2 * sigma ** 2))


def cos_sim_torch(x1, x2, eps=1e-10):
	a = torch.norm(x1)
	b = torch.norm(x2)
	if a == 0:  # for zero vector
		x1 = x1 + eps
		a = torch.norm(x1)
	if b == 0:
		x2 = x2 + eps
		b = torch.norm(x2)
	# if x1.Size < 3:
	# 	print(x1, x2, a, b, torch.dot(x1, x2) / (a * b))
	return torch.dot(x1, x2) / (a * b)


def t_distribution_torch(x1, x2):
	return 1 / (1 + (x1 - x2).pow(2).sum())


def t_distribution(x1, x2):
	return 1 / (1 + np.sum(np.square(x1 - x2)))


def gaussian_torch(x1, x2, sigma=1):
	# return torch.exp(-(x1 - x2).pow(2).sum() / (2 * sigma ** 2))
	return 1 / (1 + (x1 - x2).pow(2).sum())

def compute_j_i(X, i, sigma_i=1):
	# for each xi
	N, d = X.shape
	sum_x_ji_ = 0  # s_j|i: xi will pick xj as its neighbor, under a Gaussian centered at xi
	for k in range(N):
		if i == k: continue
		sum_x_ji_ += gaussian_torch(X[i], X[k], sigma_i).detach().numpy().item()

	return sum_x_ji_


def compute_i_j(X, j, sigma_j=1):
	# for each xj
	N, d = X.shape
	sum_x_ij_ = 0  # s_i|j: xj will pick xi as its neighbor, under a Gaussian centered at xj
	for k in range(N):
		if j == k: continue
		sum_x_ij_ += gaussian_torch(X[j], X[k], sigma_j).detach().numpy().item()

	return sum_x_ij_


def gaussian_all(X, sigma=1):
	N, d = X.shape
	s = []
	for k in range(N):
		for l in range(N):
			if k ==l: continue
			s.append(np.exp(-np.sum(np.square(X[k]-X[l]))/(2*sigma**2)))
	return s


def gaussian_xi(X, l, sigma=1):
	N, d = X.shape
	s = []
	for k in range(N):
		if k ==l: continue
		s.append(np.exp(-np.sum(np.square(X[k]-X[l]))/(2*sigma**2)))
	return s

def t_distribution_all(X):

	N, d = X.shape
	s = []
	for k in range(N):
		for l in range(N):
			if k == l : continue
			s.append(1/(1+np.sum(np.square(X[k]-X[l]))))   #
	return s



def compute_y_kl(X, net):
	N, d = X.shape
	sum_y_kl = 0
	for k in range(N):
		y_k = net(X[k])
		for l in range(N):
			if l == k: continue
			y_l = net(X[l])
			sum_y_kl += t_distribution_torch(y_k, y_l).detach().numpy().item()
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

	plt.show()
	# plt.clf()
	plt.close()


def plot_hist(ds, epoch):
	# f = os.path.join(out_dir, f'{data_name}-Euclidean.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height

	n_bins = 20
	qs = [0, 0.25, 0.5, 0.75, 1]
	d1 = np.asarray([v[0] for v in ds])
	print('original: ', np.quantile(d1, q=qs))
	cnts, bins, patches = ax[0].hist(d1, bins=n_bins)
	print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
	ax[0].set_title('Original')
	ax[0].set_ylabel('Frequency')
	ax[0].set_xlabel('$p_{ij}$')

	d2 = np.asarray([v[1] for v in ds])
	print('lower: ', np.quantile(d2, q=qs))
	cnts, bins, patches = ax[1].hist(d2, bins=n_bins)
	print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
	ax[1].set_title('Lower Space')
	ax[1].set_xlabel('$q_{ij}$')

	fig.suptitle(f'epoch: {epoch}')
	plt.tight_layout()

	plt.show()
	# plt.clf()
	plt.close()


def compute_lower_space(X, net):
	# y = net(X).detach().numpy()
	# ds = pdist(y)
	# res = np.quantile(ds, q=[0, 0.25, 0.5, 0.75, 1.0])
	# # d_u = np.max(ds)
	# # print(f'd_u: {res}')
	# return res[0], res[-1]
	y = net(X)
	ds = torch.pdist(y)
	ds = dist2_tensor(torch.min(ds), 0) + torch.max(ds, 1)


def compute_entropy(X, i, sigma):
	N, d = X.shape
	# p_ji_ = gaussian_torch(X[i], X[j], sigma)/compute_j_i(X, i, sigma_i=sigma)
	# H (P_i) = torch.sum([- p_ji_ * np.log2(p_ji_) for j in range(N) ])
	H = 0
	for j in range(N):
		if i == j: continue
		p_ji_ = gaussian_torch(X[i], X[j], sigma).detach()/compute_j_i(X, i, sigma_i=sigma)
		if p_ji_ == 0:
			H += 0
		else:
			H += -1 * p_ji_ * torch.log2(p_ji_)

	return H.numpy().item()


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
	r = 1000
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
	r = 1000
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

# def get_sigma(X, perplexity):
# 	sigmas = []
# 	N, d = X.shape
# 	for i in range(N):
# 		X_i = X[i]
# 		# find the best sigma for each X_i and given perplexity using entropy: perplexity = 2 ** H(Pi)
# 		sigma_i = compute_sigma_i(X, i, perplexity) # O(r*N**2)
# 		sigmas[tuple(X_i.detach().numpy())] = sigma_i
# 	return sigmas

def compute_P(X):
	# time complexity: O(N*(r*N**2))
	P = {}
	N, d = X.shape
	max_p = 0
	s_p = 0
	for i in range(N):
		X_i = X[i]
		for j in range(N):
			if i == j: continue
			X_j = X[j]
			# d_ij = t_distribution(X_i, X_j)
			d_ij = dist2(X_i, X_j)
			P[tuple(zip(X[i], X[j]))] = (d_ij, 0, 0)
			max_p = max(max_p, d_ij)
			s_p += d_ij

	return P, max_p, s_p

def compute_Q_ij(X, net):
	# time complexity:
	Q_ij = {}
	N, d = X.shape
	Y = net(X).detach()
	for i in range(N):
		# for each xi
		N, d = Y.shape
		sum_y_ji_ = 0  # s_j|i: xi will pick xj as its neighbor, under a Gaussian centered at xi
		for k in range(N):
			if i == k: continue
			sum_y_ji_ += t_distribution_torch(Y[k], Y[i]).detach().numpy().item()
		Q_ij[tuple(X[i].detach().numpy())] = sum_y_ji_
	return Q_ij


def main():
	out_dir = 'out'
	# data_name = '2gaussians'
	data_name = '2circles'
	# data_name = 's-curve'
	# data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	X_raw, y_raw = gen_data.gen_data(n=300, data_type=data_name, is_show=False, random_state=2)
	# X_raw = np.asarray([[0,0], [1, 3], [2, 0], [-3, 3]])
	# y_raw = np.asarray([0, 0, 1, 1])
	print(X_raw.shape, collections.Counter(y_raw))

	# # first reduce the original data to lower space to fast the latter process.
	# # std = sklearn.preprocessing.StandardScaler()
	# std = sklearn.preprocessing.MinMaxScaler()
	# std.fit(X_raw)
	# X_raw = std.transform(X_raw)
	#
	# pca = PCA(n_components=0.99, random_state=42)
	# pca.fit(X_raw)
	# print(pca.explained_variance_, pca.explained_variance_ratio_)
	# X_raw = pca.transform(X_raw)
	# print(X_raw.shape)
	#
	# # std = sklearn.preprocessing.StandardScaler()
	# std = sklearn.preprocessing.MinMaxScaler()
	# std.fit(X_raw)
	# X_raw = std.transform(X_raw)

	X, y = X_raw, y_raw
	# perplexity = min(X.shape[0]-1, 30)
	# tsne = TSNE(perplexity=perplexity, random_state=42)
	# X_ = tsne.fit_transform(X)
	#
	# plt.scatter(X_[:, 0], X_[:, 1], c=y)
	# plt.title(f'TSNE X.shape: {X_.shape}, perplexity: {perplexity}')
	# plt.show()
	#
	# umap = UMAP(n_neighbors=15, random_state=42)
	# X_ = umap.fit_transform(X_raw)
	#
	# plt.scatter(X_[:, 0], X_[:, 1], c=y_raw)
	# plt.title(f'UMAP X.shape: {X_.shape}')
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
	# plt.title(f'PCA X.shape: {X_.shape}')
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
	n_epochs = 50
	# perplexity = 20
	base = 1.
	step = base / n_epochs  # the last 5 epochs, use p_ij
	model_file = os.path.join(out_dir, f'net_ep_{n_epochs}-single-nn.pt')
	if os.path.exists(model_file):
		os.remove(model_file)

	# ds = pdist(X_raw)
	# # d_u = np.max(ds)
	# print(f'd_u: {np.quantile(ds, q=[0, 0.25, 0.5, 0.75, 1.0])}')
	if not os.path.exists(model_file):
		net = Net(in_dim=d, out_dim=out_dim)
		print(net)
		params = list(net.parameters())
		print(len(params))
		print(params[0].size())  # conv1's .weight
		# create your optimizer
		optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0)
		# optimizer = optim.SGD(net.parameters(), lr=0.01)
		# criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		# batch_size = 32
		history = {'loss': []}
		# n_neighbors = 3
		# n_neighbors = int(np.log2(n))
		n_neighbors = int(np.sqrt(n))
		# # Find the nearest neighbors for every point
		# knn = NearestNeighbors(
		# 	algorithm="auto",
		# 	n_neighbors=n_neighbors,
		# )
		# knn.fit(X_raw)
		# distances_nn = knn.kneighbors_graph(mode="distance")
		# sigmas = {}
		# sum_x = {}
		# P, max_p,s_p = compute_P(X_raw)
		centroids = {}
		for epoch in range(n_epochs + 1):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			N, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)

			knn = NearestNeighbors(
				algorithm="auto",
				n_neighbors=n_neighbors,
			)
			knn.fit(X)
			# dists, indices = knn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)
			dists, indices = knn.radius_neighbors(X,radius=0.5, return_distance=True)
			X = torch.from_numpy(X).float()

			L = 0
			optimizer.zero_grad()  # zero the gradient buffers
			loss = 0
			losses = []

			ds = []

			Y = net(X)
			for i in range(N):
				l = 0
				for j, idx in enumerate(indices[i]):
					if i == idx: continue
					l += (1/dists[i][j] * Y[i] - Y[idx]).pow(2).sum()
					# l += (1 / (dists[i][j] / max(dists[i])) * Y[i] - Y[idx]).pow(2).sum()
				# print(j+1)
			loss += (l / (j+1)) + 0.2 * 1/(Y[i].pow(2).sum())       # || Y[i] || = 1

			loss.backward()
			optimizer.step()  # Does the update

			losses.append(loss)
			L += loss.item()
			print(f'{epoch + 1}/{n_epochs}, loss: {L},  base: {base}')
			history['loss'].append(L)

			# plot_xy(X, y, net, epoch
			if epoch > 0 and epoch % 10 == 0:
				base = base - step if base - step > 1 else 1
				# plot_hist(ds, epoch)

			if epoch > 0 and epoch % 10 == 0:
				print(f'*** lr:{scheduler.get_lr()}')
				scheduler.step()  # adjust learning rate
			# plot_xy(X, y, net, epoch)

		f = os.path.join(out_dir, f'{data_name}-loss.png')
		plt.plot(history['loss'][-100:])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

		# save model
		with open(model_file, 'wb') as f:
			torch.save(net, f)
	else:
		net = torch.load(model_file)

	# X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
	# X = torch.from_numpy(X).float()
	# N, d = X.shape
	# s_i = 0
	# sum_y_ij = compute_y_kl(X, net)  # we need to recompute because the net is updated
	# for i in range(N):
	# 	y_i = net(X[i])
	# 	for j in range(i + 1, N):
	# 		k1 = tuple(zip(X[i].detach().numpy(), X[j].detach().numpy()))
	# 		p_ji_, sigma_i, sum_x_ji_ = P[k1]
	# 		k2 = tuple(zip(X[j].detach().numpy(), X[i].detach().numpy()))
	# 		p_ij_, sigma_j, sum_x_ij_ = P[k2]
	#
	# 		p_ij = (p_ji_ + p_ij_) / (2 * N)
	#
	# 		y_j = net(X[j])
	# 		q_ij = t_distribution_torch(y_i, y_j).detach().numpy().item() / sum_y_ij
	#
	# 		# print(i, j, (dist2_tensor(X[i], X[j]) / mn) ** power, dist2_tensor(y_i, y_j).pow(1 / 2))
	# 		# print(i, j, (gaussian_torch(X[i], X[j],  sigma=sigma)-mn)/(mx-mn),(t_distribution_torch(y_i, y_j)-mn2)/(mx2-mn2))
	# 		print(i, j, p_ij, q_ij, np.log2(p_ij/q_ij))
	# 		# s_i += dist2_tensor((dist2_tensor(X[i], X[j]) / mn) ** power, dist2_tensor(y_i, y_j).pow(1 / 2)) * 2
	# print(f's_i: {s_i}')

	f = os.path.join(out_dir, f'{data_name}-Euclidean.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height

	m = N // N  # N//5
	indices = range(0, N, m)
	X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
	X, y = X[indices, :], y[indices]
	mn = np.min(X)
	mx = np.max(X)
	if len(set(y)) != 2:
		ax[0].scatter(X[:, 0], X[:, 1], c=y)
	else:
		ax[0].scatter(X[:, 0], X[:, 1], c=['g' if v == 0 else 'r' for v in y])
	if X.shape[1] < 2:
		for i in range(N):
			txt = f'{i}:{X[i]}'
			ax[0].annotate(txt, (X[i, 0], X[i, 1]))
	ax[0].set_title(f'Original Space')
	# ax[0].set_xlim(mn, mx)
	# ax[0].set_ylim(mn, mx)
	# ax[0].set_aspect('equal', 'box')
	# print(mn, mx)

	X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
	X = torch.from_numpy(X).float()
	O = net(X)
	O = O.detach().numpy()
	mn, mx = np.min(O), np.max(O)
	if len(set(y)) != 2:
		ax[1].scatter(O[:, 0], O[:, 1], c=y)
	else:
		ax[1].scatter(O[:, 0], O[:, 1], c=['g' if v == 0 else 'r' for v in y])
	if X.shape[1] < 2:
		for i in range(N):
			txt = f'{i}:{X[i].detach().numpy()}->{O[i]}'
			ax[1].annotate(txt, (O[i, 0], O[i, 1]), fontsize=6)
	ax[1].set_title('Lower space')
	# ax[1].set_xlim(mn, mx)
	# ax[1].set_ylim(mn, mx)
	# ax[1].set_aspect('equal', 'box')

	fig.suptitle(f'X.shape: {X.shape}. Euclidean, n_neighbors={n_neighbors}' )
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close()

	print(datetime.datetime.now())


if __name__ == '__main__':
	main()
