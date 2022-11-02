"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import collections
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import ExponentialLR
from umap import UMAP

from datasets import gen_data
from utils.common import check_path

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

np.set_printoptions(precision=3)


class Net(nn.Module):

	def __init__(self, in_dim, out_dim=2):
		super(Net, self).__init__()
		# an affine operation: y = Wx + b
		dim = max(in_dim, out_dim * 8*2)
		self.fc1 = nn.Linear(in_dim, dim//4)
		self.fc2 = nn.Linear(dim//4,dim//8)
		self.fc20 = nn.Linear(dim//8, dim//8)
		self.fc3 = nn.Linear(dim//8, out_dim)

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		# x = F.leaky_relu(self.fc20(x))
		# x = F.leaky_relu(self.fc20(x))
		# x = F.leaky_relu(self.fc20(x))
		x = F.leaky_relu(self.fc20(x))
		# x = F.sigmoid(self.fc3(x))
		x = F.softmax(self.fc3(x))
		# x = self.fc3(x)
		return x


def dist2(x1, x2):
	return np.sum(np.square(x1 - x2))


def compute_Xy(X):
	n, d = X.shape
	X_ = []
	y_ = []
	for i in range(n):
		# for j in range(n):
		for j in range(i, n):
			X_.append(np.concatenate([X[i], X[j]]))
			y_.append(dist2(X[i], X[j]))  # y_.append(dist2(X[i], X[j])/d)
	# print('test')

	return np.asarray(X_), np.asarray(y_)


#
# def compute_Xy2(X):
# 	n, d = X.shape
# 	# X_ = []
# 	# y_ = []
# 	# for i in range(n):
# 	# 	X_.append(np.concatenate([x1, X2[i]]))
# 	# 	y_.append(dist2(x1, X2[i]) / d)
# 	X_ = []
# 	y_ = []
# 	for i in range(n-1):
# 		X_.append(np.concatenate([X[i], X[i+1]]))
# 		y_.append(dist2(X[i], X[i+1]) / d)
#
# 	return np.asarray(X_), np.asarray(y_)


def compute_Xy2(X1, X):
	n, d = X.shape
	# X_ = []
	# y_ = []
	# for i in range(n):
	# 	X_.append(np.concatenate([x1, X2[i]]))
	# 	y_.append(dist2(x1, X2[i]) / d)
	X_ = []
	y_ = []
	for i in range(n):
		X_.append(np.concatenate([X1, X[i]]))
		# y_.append(dist2(X1, X[i])/d)
		y_.append(dist2(X1, X[i]))

	return np.asarray(X_), np.asarray(y_)


def dist2_tensor(X1, X2):
	return (X1 - X2).pow(2).sum(axis=1)


# return (X1 - X2).abs().sum(axis=1)


def compute_d2(X, in_dim, out, out_dim):
	# Y  = torch.unique(X[:, :in_dim]) # unique X
	Y = set([str(v.numpy()) for v in X[:, :in_dim]])
	s = 0
	cnt = 0
	for x_ in Y:
		# print(len(Y), x_)
		# indices = np.equal(X[:, :in_dim], x_)
		indices = [i for i in range(X.shape[0]) if str(X[i][:in_dim].numpy()) == x_]
		cnt += len(indices)
		# print(len(indices), cnt, f'{cnt/X.shape[0] * 100:.2f}', X.shape[0])
		O1 = out[indices][:, :out_dim]
		m, _ = O1.shape
		for i in range(0, m):
			for j in range(i + 1, m):
				s += (O1[i] - O1[j]).pow(2).sum()
	return s


def cos_sim(x1, x2):
	return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# return np.dot(x1, x2)

def cos_sim_torch(x1, x2):
	return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))


def main():
	X_raw, y_raw = gen_data.gen_data(n=100, data_type='2gaussians', is_show=False, random_state=42)
	print(X_raw.shape, collections.Counter(y_raw))
	plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw)
	plt.title('X')
	plt.show()

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
	k = 2
	n_epochs = 200
	model_file = f'cluster_ep_{n_epochs}-n5.pt'
	if os.path.exists(model_file):
		os.remove(model_file)

	ds = pdist(X_raw)
	d_u = np.max(ds)
	print(f'd_u: {d_u}')
	if not os.path.exists(model_file):
		net = Net(in_dim=d, out_dim= k * d)
		print(net)
		params = list(net.parameters())
		print(len(params))
		print(params[0].size())  # conv1's .weight
		# create your optimizer
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		# optimizer = optim.SGD(net.parameters(), lr=0.01)
		criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		batch_size =  10  #X_raw.shape[0]
		history = {'loss': []}
		for epoch in range(n_epochs):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			n, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
			losses = []
			L = 0
			seen = {}
			seen_cos = {}
			for i in range(0, n, batch_size):
				X1 = X[i:i + batch_size, :]
				m, d = X1.shape
				loss = 0
				optimizer.zero_grad()  # zero the gradient buffers
				X1 = torch.from_numpy(X1).float()
				out = net(X1)

				ds = torch.zeros((m, k))
				for t in range(k):
					ds[:, t] = (out[:, t*d: (t+1)*d] - X1).pow(2).sum(axis=1)
				labels = torch.argmin(ds, axis=1)
				for t in range(k):
					mask = labels == t
					if sum(mask) > 0:
						mean_ = torch.mean(out[mask, t*d:(t+1)*d], axis=0)
						# loss += (out[mask, t*d:(t+1)*d] - mean_).pow(2).sum()   #+ (out[mask][0] - out[mask]).pow(2).sum()
						mean2_ = torch.mean(X1[mask], axis=0)
						loss += (X1[mask]- mean_).pow(2).sum()/sum(mask)
						loss += (mean2_- mean_).pow(2).sum()
						loss + (out[~mask, t*d:(t+1)*d]-0).pow(2).sum()
				# for j in range(m):
				# 	out = net(X1[j])
				# 	idx = 0
				# 	tmp0 = torch.inf
				# 	for t in range(k):
				# 		tmp = (out[t*d: (t+1)*d] - X1[j]).pow(2).sum()
				# 		if tmp < tmp0:
				# 			tmp0 = tmp
				# 			idx = t
				# 			# mp[idx].append(out[t*d: (t+1)*d])
				#
				# 	loss += tmp0
				# 	# loss += torch.min([(out[t*d: (t+1)*d] - X1[j].pow(2).sum())for t in range(k)])

				loss.backward()
				optimizer.step()  # Does the update
				losses.append(loss.item())
				L += loss.item()

			X1 = torch.from_numpy(X).float()
			out = net(X1)
			ds = torch.zeros((X1.shape[0], k))
			for t in range(k):
				ds[:, t] = (out[:, t * d: (t + 1) * d] - X1).pow(2).sum(axis=1)
			labels = torch.argmin(ds, axis=1)
			centroids = []
			centroids2 = []
			cnts = []
			for t in range(k):
				mask = labels == t
				centroids.append(list(torch.mean(out[mask, t*d:(t+1)*d], axis=0).detach().numpy()))
				centroids2.append(list(torch.mean(X1[mask], axis=0).detach().numpy()))
				cnts.append(sum(mask))
			print(f'{epoch + 1}/{n_epochs}, loss: {L}, centroids: {centroids}, cnts: {cnts}, centroids2: {centroids2}')
			history['loss'].append(L)

		# if epoch % 50==0:
		# 	print(scheduler.get_lr())
		# 	scheduler.step()  # adjust learning rate

		plt.plot(history['loss'])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.show()

		# save model
		with open(model_file, 'wb') as f:
			torch.save(net, f)
	else:
		net = torch.load(model_file)




if __name__ == '__main__':
	main()
