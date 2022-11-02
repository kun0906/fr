""" A neural network-based dimensionality reduction method

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
# Author: Kun.bj@outlook.com

import collections
import copy
import os
import random
import datetime

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
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
		dim = max(in_dim, out_dim * 8 * 100)
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
	return (X1 - X2).pow(2).sum()


def cos_sim_torch(x1, x2):
	a = torch.norm(x1)
	b = torch.norm(x2)
	if a == 0:  # for zero vector
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
	print([f'{bins[i:i+2]}:{cnts[i]}' for i in range(n_bins)])
	ax[0].set_title('Original')
	ax[0].set_ylabel('Frequency')
	ax[0].set_xlabel('$||X_i - X_j||$')

	d2 = np.asarray([v[1] for v in ds])
	print('lower: ', np.quantile(d2, q=qs))
	cnts, bins, patches = ax[1].hist(d2, bins=n_bins)
	print([f'{bins[i:i+2]}:{cnts[i]}' for i in range(n_bins)])
	ax[1].set_title('Lower Space')
	ax[1].set_xlabel('$||y_i - y_j||$')

	fig.suptitle(f'epoch: {epoch}')
	plt.tight_layout()

	plt.show()
	# plt.clf()
	plt.close()



def main():
	out_dir = 'out'
	data_name = '2gaussians'
	data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	X_raw, y_raw = gen_data.gen_data(n=2, data_type=data_name, is_show=False, random_state=42)
	# X_raw = np.asarray([[0,0], [1, 3], [2, 0], [-3, 3]])
	# y_raw = np.asarray([0, 0, 1, 1])
	print(X_raw.shape, collections.Counter(y_raw))

	# first reduce the original data to lower space to fast the latter process.
	std = sklearn.preprocessing.StandardScaler()
	std.fit(X_raw)
	X_raw = std.transform(X_raw)

	pca = PCA(n_components=0.99, random_state=42)
	pca.fit(X_raw)
	print(pca.explained_variance_, pca.explained_variance_ratio_)
	X_raw = pca.transform(X_raw)
	print(X_raw.shape)

	std = sklearn.preprocessing.StandardScaler()
	std.fit(X_raw)
	X_raw = std.transform(X_raw)


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
		optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.5)
		# optimizer = optim.SGD(net.parameters(), lr=0.01)
		criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		batch_size = 32
		history = {'loss': []}
		n_neighbors = 5
		# Find the nearest neighbors for every point
		knn = NearestNeighbors(
			algorithm="auto",
			n_neighbors=n_neighbors,
		)
		knn.fit(X_raw)
		# distances_nn = knn.kneighbors_graph(mode="distance")

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
			if epoch == 0:
				mn = 1
				mn2 = 1
				s1 = 0
				s2 = 0
			else:
				mn = pre_mn.item()
				mn2 = pre_mn2.item()
				s1 = pre_s1.item()
				s2 = pre_s2.item()
			pre_mn = 0
			pre_mn2 = 0
			pre_s1 = 0
			pre_s2 = 0
			s_i = 0
			l_i = 0
			ds = []
			for i in range(N):
				X_i = X[i]
				y_i = net(X_i)
				# seen[tuple(X[i])] = y_i
				# indices = knn.kneighbors(X_i.detach().numpy().reshape(1, -1), n_neighbors=n_neighbors, return_distance=False)[0]
				# X2 = torch.from_numpy(X_raw[indices]).float()
				X2 = X
				for X_j in X2:
					if torch.all(X_j == X_i): continue
					y_j = net(X_j)
					pre_mn = max(pre_mn, dist2_tensor(X_i, X_j).pow(1/2).detach().numpy())
					pre_mn2 = max(pre_mn2, dist2_tensor(y_i, y_j).pow(1 / 2).detach().numpy())
					pre_s1 += dist2_tensor(X_i, X_j).pow(1 / 2).detach().numpy()
					pre_s2 += dist2_tensor(y_i, y_j).pow(1 / 2).detach().numpy()
					# a = dist2_tensor(torch.exp(dist2_tensor(X_i, X_j).pow(1/2)/mn)-1, dist2_tensor(y_i, y_j).pow(1/2))
					# a = dist2_tensor(2.7**(dist2_tensor(X_i, X_j).pow(1 / 2) / mn) - 1,
					#                  dist2_tensor(y_i, y_j).pow(1 / 2))
					power = 5
					a = (dist2_tensor(X_i, X_j).pow(1 / 2) / mn).detach().numpy().item()
					d_ij = (dist2_tensor(X_i, X_j).pow(1 / 2) / mn) ** power    # normalized to [0, 1]
					# d_ij = torch.exp(dist2_tensor(X_i, X_j).pow(1 / 2) / mn) # sum all d_ij is 1
					d_ij2 = dist2_tensor(y_i, y_j).pow(1 / 2) / mn2  #/ s2           # expected to [0, 1]
					w_ij = 1
					# w_ij = 1/(a**2)        # decreased function
					# w_ij = np.exp(-a*5)     # decreased function
					# w_ij = np.abs(a-0.5)**1   # bowl shape function
					# diff = w_ij * dist2_tensor(d_ij, d_ij2)
					diff = w_ij * (torch.log2(d_ij/d_ij2)).abs() # the larger the d_ij, the smaller for the loss.
					ds.append((d_ij.detach().numpy(), d_ij2.detach().numpy()))
					# w_ij = diff.detach()
					# s_i += diff.detach()
					w_ij = s_i = 1
					# a = (dist2_tensor(X_i, X_j).pow(1 / 2) / mn)**5
					# b =  dist2_tensor(y_i, y_j).pow(1 / 2)
					# print(i, torch.log2(d_ij/d_ij2))
					# diff = d_ij * (torch.log2(d_ij/d_ij2)).abs()
					# a = dist2_tensor(X_i, X_j) / dist2_tensor(y_i, y_j)
					l_i += w_ij * diff      # weighted loss for each X_i
					# loss += dist2_tensor(cos_sim_torch(X_i, X_j), cos_sim_torch(y_i, y_j))
					alpha = 1
			if epoch == 0: continue
			loss += l_i / s_i
			# print(i, s_i, l_i, loss)
			losses.append(loss)
				# loss += dist2_tensor(dist2_tensor(X[i], X[j]), dist2_tensor(y_i, y_j))  + alpha * dist2_tensor(cos_sim_torch(X[i], X[j]), cos_sim_torch(y_i, y_j))
				# print(i, j, dist2_tensor(X[i], X[j]), dist2_tensor(y_i, y_j), loss)
			# print(i, sum_p_ji)
			if epoch % 50 == 0:
				plot_hist(ds, epoch)
			# loss = 0
			# tmp = sorted(losses, reverse=True)[:2]
			# for v in tmp:
			# 	loss += v
			# plot_xy(X, y, net, epoch)
			# loss = loss / (N*N)
			loss.backward()
			optimizer.step()  # Does the update

			L += loss.item()
			print(f'{epoch + 1}/{n_epochs}, loss: {L}, mn: {mn}, mn2: {mn2}, s1: {s1}, s2:{s2}')
			history['loss'].append(L)
			if epoch % 50 == 0:
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

	X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
	X  = torch.from_numpy(X).float()
	N, d = X.shape
	s_i = 0
	for i in range(N):
		for j in range(i+1, N):
			y_i = net(X[i])
			y_j = net(X[j])
			print(i, j, (dist2_tensor(X[i], X[j]).pow(1/2)/mn)**power, dist2_tensor(y_i, y_j).pow(1/2))
			s_i += dist2_tensor((dist2_tensor(X[i], X[j]).pow(1/2)/mn)**power, dist2_tensor(y_i, y_j).pow(1/2)) * 2
	print(f's_i: {s_i}')

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
	if X.shape[1] < 3:
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
	mn,mx = np.min(O), np.max(O)
	if len(set(y)) != 2:
		ax[1].scatter(O[:, 0], O[:, 1], c=y)
	else:
		ax[1].scatter(O[:, 0], O[:, 1], c=['g' if v == 0 else 'r' for v in y])
	if X.shape[1] < 3:
		for i in range(N):
			txt = f'{i}:{X[i].detach().numpy()}->{O[i]}'
			ax[1].annotate(txt, (O[i, 0], O[i, 1]), fontsize = 6)
	ax[1].set_title('Lower space')
	# ax[1].set_xlim(mn, mx)
	# ax[1].set_ylim(mn, mx)
	# ax[1].set_aspect('equal', 'box')

	fig.suptitle('Euclidean')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close()

	print(datetime.datetime.now())


if __name__ == '__main__':
	main()
