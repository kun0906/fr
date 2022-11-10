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
import torch.optim as optim
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from torch.optim.lr_scheduler import ExponentialLR

from datasets import gen_data
from inc_tsne import _joint_probabilities

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

np.set_printoptions(precision=3)

print(datetime.datetime.now())


class AutoEncoder(nn.Module):
	def __init__(self, in_dim, out_dim=2):
		super(AutoEncoder, self).__init__()
		h1 = max(in_dim // 2, out_dim)
		h2 = max(h1 // 2, out_dim)
		act = nn.LeakyReLU(True)
		# act = nn.Tanh()
		self.encoder = nn.Sequential(
			nn.Linear(in_dim, h1, bias=False),
			act,
			nn.Linear(h1, h2, bias=False),
			act,
			nn.Linear(h2, out_dim, bias=False),
		)
		self.decoder = nn.Sequential(
			nn.Linear(out_dim, h2, bias=False),
			act,
			nn.Linear(h2, h1, bias=False),
			act,
			nn.Linear(h1, in_dim, bias=False),
			# nn.Sigmoid()
		)
		# encoder and decoder share the weights and bias
		# https://discuss.pytorch.org/t/how-to-share-weights-between-two-layers/55541
		i = 0
		while i < 5:
			self.decoder[4 - i].weight = nn.Parameter(self.encoder[i].weight.t())
			# self.decoder[4-i].bias = nn.Parameter(self.encoder[i].bias)
			i += 2

	def forward(self, x):
		x1 = self.encoder(x)
		x2 = self.decoder(x1)
		return x1, x2


def dist2_tensor(X1, X2):
	# return (X1 - X2).pow(2).sum()
	# return (X1 - X2).abs().sum()
	k = 1
	return (X1 - X2).abs().pow(k).sum().pow(1 / k)


def dist2(X1, X2):
	# return np.sum(np.square(X1 - X2))
	k = 1
	return np.sum(np.abs(X1 - X2) ** (k)) ** (1 / k)


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
			if k == l: continue
			s.append(np.exp(-np.sum(np.square(X[k] - X[l])) / (2 * sigma ** 2)))
	return s


def gaussian_xi(X, l, sigma=1):
	N, d = X.shape
	s = []
	for k in range(N):
		if k == l: continue
		s.append(np.exp(-np.sum(np.square(X[k] - X[l])) / (2 * sigma ** 2)))
	return s


def t_distribution_all(X):
	N, d = X.shape
	s = []
	for k in range(N):
		for l in range(N):
			if k == l: continue
			s.append(1 / (1 + np.sum(np.square(X[k] - X[l]))))  #
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


def plot_grad(parameters, epoch):
	# f = os.path.join(out_dir, f'{data_name}-Euclidean.png')

	n = len(parameters)
	nrows, ncols = n // 2, 4
	fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15), )  # width, height
	n_bins = 20
	qs = [0, 0.25, 0.5, 0.75, 1]
	for i in range(nrows):
		# weights
		ax = axes[i][0]
		d1 = parameters[i].detach().numpy().flatten()
		# print('original: ', np.quantile(d1, q=qs))
		cnts, bins, patches = ax.hist(d1, bins=n_bins)
		print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
		# ax.set_title('Weights')
		ax.set_ylabel('Frequency')
		ax.set_xlabel('Weights')

		# weights.grads
		ax = axes[i][1]
		d1 = parameters[i].grad.detach().numpy().flatten()
		# print('original: ', np.quantile(d1, q=qs))
		cnts, bins, patches = ax.hist(d1, bins=n_bins)
		# print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
		# ax[0].set_title('Weights')
		ax.set_ylabel('Frequency')
		ax.set_xlabel('Weights.Grads')

		# Weights
		ax = axes[i][2]
		d1 = parameters[n - 1 - i].detach().numpy().flatten()
		# print('original: ', np.quantile(d1, q=qs))
		cnts, bins, patches = ax.hist(d1, bins=n_bins)
		# print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
		# ax.set_title('Weights')
		ax.set_ylabel('Frequency')
		ax.set_xlabel('Weights(decoder)')

		# Weights.grads
		ax = axes[i][3]
		d1 = parameters[n - 1 - i].grad.detach().numpy().flatten()
		# print('original: ', np.quantile(d1, q=qs))
		cnts, bins, patches = ax.hist(d1, bins=n_bins)
		# print([f'{bins[i:i + 2]}:{cnts[i]}' for i in range(n_bins)])
		# ax[0].set_title('Weights')
		ax.set_ylabel('Frequency')
		ax.set_xlabel('Weights.Grads(decoder)')

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


def compute_P(X, desired_perplexity=30):
	distances = pairwise_distances(X, squared=True)
	P, sum_P, conditional_P, C_S_Pi, Beta = _joint_probabilities(distances, desired_perplexity, verbose=0)
	# # time complexity: O(N*(r*N**2))
	# P = {}
	# N, d = X.shape
	# max_p = 0
	# s_p = 0
	# for i in range(N):
	# 	X_i = X[i]
	# 	for j in range(N):
	# 		if i == j: continue
	# 		X_j = X[j]
	# 		# d_ij = t_distribution(X_i, X_j)
	# 		d_ij = dist2(X_i, X_j)
	# 		P[tuple(zip(X[i], X[j]))] = (d_ij, 0, 0)
	# 		max_p = max(max_p, d_ij)
	# 		s_p += d_ij
	#
	# return P, max_p, s_p
	return squareform(P), sum_P, Beta


def compute_Q_ij(Y):
	# time complexity:
	N, d = Y.shape
	Q_ij = torch.zeros(((N, N)))
	sum_y = 0  #
	for k in range(N):
		for l in range(N):
			if k == l: continue
			sum_y += t_distribution_torch(Y[k], Y[l]).detach().numpy().item()

	for i in range(N):
		for j in range(N):
			if i == j: continue
			Q_ij[i][j] = t_distribution_torch(Y[j], Y[i]) / sum_y
	return Q_ij


def main():
	out_dir = 'out'
	# data_name = '2gaussians'
	# data_name = '2circles'
	# data_name = 's-curve'
	data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	X_raw, y_raw = gen_data.gen_data(n=10, data_type=data_name, is_show=False, random_state=2)
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
	perplexity = min(X.shape[0] - 1, 30)
	tsne = TSNE(perplexity=perplexity, random_state=42)
	X_ = tsne.fit_transform(X)

	plt.scatter(X_[:, 0], X_[:, 1], c=y)
	plt.title(f'TSNE X.shape: {X_.shape}, perplexity: {perplexity}')
	plt.show()
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
	n_epochs = 100
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
		net = AutoEncoder(in_dim=d, out_dim=out_dim)
		print(net)
		params = list(net.parameters())
		print(len(params))
		print(params[0].size())  # conv1's .weight
		# create your optimizer
		optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=0.1)
		# optimizer = optim.SGD(net.parameters(), lr=0.01)
		# criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		# batch_size = 32
		history = {'loss': []}
		# n_neighbors = 3
		# n_neighbors = int(np.log2(n))
		n_neighbors = n
		# # Find the nearest neighbors for every point
		# knn = NearestNeighbors(
		# 	algorithm="auto",
		# 	n_neighbors=n_neighbors,
		# )
		# knn.fit(X_raw)
		# # distances_nn = knn.kneighbors_graph(mode="distance")
		# sigmas = {}
		# sum_x = {}
		# P, sum_P, Beta = compute_P(X_raw, desired_perplexity=30)
		# P = torch.from_numpy(P)
		centroids = {}
		for epoch in range(n_epochs + 1):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			N, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
			X = torch.from_numpy(X).float()

			L = 0
			optimizer.zero_grad()  # zero the gradient buffers
			loss = 0
			losses = []

			ds = []

			Y, X_pred = net(X)
			loss1 = (X - X_pred).pow(2).sum()
			# Q = compute_Q_ij(Y)
			# l = P * torch.log2(P/Q)
			# l.fill_diagonal_(0)
			# l2 = torch.sum(l.abs())
			# # loss += 10 * l2
			# print(epoch, l1, l2)
			labels = set(y)
			loss2 = 0
			for l1 in labels:
				indices1 = (y == l1)
				if sum(indices1) == 0: continue

				# loss += torch.sum(torch.pairwise_distance(Y[indices1]))
				# mu_X = torch.mean(X[indices1], axis=0)
				mu1_Y = torch.mean(Y[indices1], axis=0)
				n1 = sum(indices1)
				loss2 += (Y[indices1] - mu1_Y).pow(2).sum()
				# p = q = 0
				# for x_, y_ in zip(X[indices1], Y[indices1]):
				# 	p += dist2_tensor(x_, mu_X)
				# 	q += dist2_tensor(y_, mu_Y)
				# 	# loss += torch.log((p / s_p) / (q / s_q)).abs()
				s = set()
				for l2 in labels:  # different clusters
					indices2 = (y == l2)
					if sum(indices2) == 0: continue
					if l1 == l2 or l2 in s: continue
					# s.add(l2)
					n2 = sum(indices2)
					mu2_Y = torch.mean(Y[indices2], axis=0)
					loss2 += (Y[indices2] - mu2_Y).pow(2).sum()
					# loss += 1/torch.sum(torch.pairwise_distance(Y[indices1], Y[indices2]))    # noise impacts
					loss2 += n1 * n2 * 1 / dist2_tensor(mu1_Y, mu2_Y)
			# loss += torch.sum(1/torch.cdist(Y[indices1], Y[indices2]))  # more weights on small distances
			# loss += 1 / dist2_tensor(torch.mean(Y[indices1], axis=0), torch.mean(Y[indices2], axis=0))

			alpha = 0.0
			loss = alpha * loss1 + (1 - alpha) * loss2
			#
			# 		# for x_, y_ in zip(X[indices], Y[indices]):
			# 		# 	p += dist2_tensor(x_, mu_X)
			# 		# 	q += dist2_tensor(y_, mu_Y)
			#
			# 		p = 0
			# 		q = 0
			# 		# same clusters 1
			# 		a = 1
			# 		b = 1
			# 		for x1, y1 in zip(X[indices1], Y[indices1]):
			# 			for x2, y2 in zip(X[indices1], Y[indices1]):
			# 				p += dist2_tensor(x1, x2)
			# 				q += a * dist2_tensor(y1, y2)
			# 		# loss += dist2_tensor(p / s_p, b * q / s_q)
			# 		# print('1: ', p, s_p, s_q, q, loss)
			# 		# loss += torch.log((p/s_p)/(q/s_q)).abs()
			# 		# same clusters 2
			# 		p = 0
			# 		q = 0
			# 		for x1, y1 in zip(X[indices2], Y[indices2]):
			# 			for x2, y2 in zip(X[indices2], Y[indices2]):
			# 				p += dist2_tensor(x1, x2)
			# 				q += a * dist2_tensor(y1, y2)
			# 		# loss += dist2_tensor(p / s_p, b * q / s_q)
			# 		# print('2: ', p, s_p, s_q, q, loss)
			# 		# loss += torch.log((p/s_p)/(q/s_q)).abs()
			# 		# different clusters
			# 		p = 0
			# 		q = 0
			# 		for x1, y1 in zip(X[indices1], Y[indices1]):
			# 			for x2, y2 in zip(X[indices2], Y[indices2]):
			# 				d1 = dist2_tensor(x1, x2)
			# 				w1 = d1
			# 				p += w1
			# 				d2 = dist2_tensor(y1, y2)
			# 				w2 = d2
			# 				q += w2
			# 		# mu11 = torch.mean(X[indices1], axis=0)
			# 		# mu12 = torch.mean(X[indices2], axis=0)
			# 		# mu21 = torch.mean(Y[indices1], axis=0)
			# 		# mu22 = torch.mean(Y[indices2], axis=0)
			# 		# m1 = len(indices1)
			# 		# m2 = len(indices2)
			# 		# # p = p/(m1*m1+m2*m2)
			# 		# # q = q/(m1*m1+m2*m2)
			# 		# p += dist2_tensor(mu11, mu12)
			# 		# q += 1*dist2_tensor(mu21, mu22)
			#
			# 		# loss += dist2_tensor(p/s_p, b* q/s_q)
			# 		# loss += dist2_tensor(p, b * q )
			# 		# loss += torch.log((p/s_p)/(q/s_q)).abs()
			# 		# loss += q
			# 		# update
			# 		# print('3: ', p, s_p, s_q, q, loss)
			p = q = torch.zeros((1, 1))
			ds.append((p.detach().numpy().item(), q.detach().numpy().item()))
			if loss == 0: continue

			loss.backward()  # Compute the gradient for each parameter: dloss/dw
			optimizer.step()  # Does the update: w = w - \eta * dloss/dw

			losses.append(loss)
			L += loss.item()
			print(f'{epoch + 1}/{n_epochs}, loss: {L},  base: {base}')
			history['loss'].append(L)

			# plot_xy(X, y, net, epoch
			if epoch > 0 and epoch % 10 == 0:
				base = base - step if base - step > 1 else 1
				# plot_hist(ds, epoch)
				# plot_grad(list(net.parameters()), epoch)

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
	O, _ = net(X)
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

	fig.suptitle(f'X.shape: {X.shape}. Euclidean, n_neighbors={n_neighbors}')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close()

	print(datetime.datetime.now())


if __name__ == '__main__':
	main()
