"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import collections

import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import gen_data
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class Net(nn.Module):

	def __init__(self, in_dim, out_dim=2):
		super(Net, self).__init__()
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(in_dim, 10)  # 5*5 from image dimension
		self.fc2 = nn.Linear(10, 5)
		self.fc3 = nn.Linear(5, 2 * out_dim)

	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		x = self.fc3(x)
		return x


def dist2(x1, x2):
	return np.sum(np.square(x1 - x2))


def compute_Xy(X):
	n, d = X.shape
	X_ = []
	y_ = []
	for i in range(n):
		for j in range(n):
		# for j in range(i + 1, n):
			X_.append(np.concatenate([X[i], X[j]]))
			y_.append(dist2(X[i], X[j]) / d)
		# print('test')

	return np.asarray(X_), np.asarray(y_)


def compute_Xy2(X):
	n, d = X.shape
	# X_ = []
	# y_ = []
	# for i in range(n):
	# 	X_.append(np.concatenate([x1, X2[i]]))
	# 	y_.append(dist2(x1, X2[i]) / d)
	X_ = []
	y_ = []
	for i in range(n-1):
		X_.append(np.concatenate([X[i], X[i+1]]))
		y_.append(dist2(X[i], X[i+1]) / d)

	return np.asarray(X_), np.asarray(y_)


def main():
	X, y = gen_data.gen_data(n=5000, is_show=True, random_state=42)
	print(X.shape, collections.Counter(y))
	n, d = X.shape
	out_dim = d
	net = Net(in_dim=2 * d, out_dim=out_dim)
	print(net)
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight
	# create your optimizer
	optimizer = optim.Adam(net.parameters(), lr=0.01)
	criterion = nn.MSELoss()
	# criterion = nn.CrossEntropyLoss()
	n_epochs = 100
	batch_size = 32
	history = {'loss': []}
	for epoch in range(n_epochs):
		# shuffle X, y
		n, d = X.shape
		X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
		losses = []
		for i in range(0, n, batch_size):
			X_, y_ = compute_Xy(X[i:i + batch_size, :])
			X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
			# in your training loop:
			optimizer.zero_grad()  # zero the gradient buffers
			out = net(X_)
			out = (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(
				axis=1) / out_dim  # squared euclidean distance of X1 and X2
			loss = criterion(out, y_)
			loss.backward()
			optimizer.step()  # Does the update
			losses.append(loss.item())

		print(f'{epoch + 1}/{n_epochs}, loss: {loss}')
		history['loss'].append(loss.item())

	import matplotlib.pyplot as plt
	plt.plot(history['loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()

	# for all pairs
	X_, y_ = compute_Xy(X)
	X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	out = net(X_)
	# print(out)
	# d - out_d
	dist_ =  (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(axis=1).detach().numpy() / out_dim
	y_ = y_.detach().numpy()
	print(y_ - dist_)
	print(dist_.shape, y_.shape)
	print([(y_[i*n+i], dist_[i*n+i]) for i in range(n)])

	# for testing
	y2 = y[:n-1]  # exclude the first point
	X_, y_ = compute_Xy2(X[0, :], X[1:, :])  # X1 and the rest data
	# X_, y_ = compute_Xy2(X)  # X1 and the rest data
	X_, y_ = X_[:n, :], y_[:n]  # X1 to other points
	X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	out = net(X_)
	X2 = out[:, out_dim:].detach().numpy()
	print(X2.shape)
	# y2 =
	# plt.scatter(X2, y2)
	plt.scatter(X2[:, 0], X2[:, 1], c=y2)
	plt.show()


if __name__ == '__main__':
	main()
