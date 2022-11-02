"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import collections
import random

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import gen_data

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class Net(nn.Module):

	def __init__(self, in_dim, out_dim=2):
		super(Net, self).__init__()
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(in_dim, 10)
		self.fc2 = nn.Linear(10, 5)
		self.fc20 = nn.Linear(5, 5)
		self.fc3 = nn.Linear(5, out_dim)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = F.tanh(self.fc20(x))
		x = F.tanh(self.fc20(x))
		x = F.tanh(self.fc20(x))
		x = F.tanh(self.fc20(x))
		x = self.fc3(x)
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
	# return (X1 - X2).pow(2).sum(axis=1)
	return (X1 - X2).abs().sum(axis=1)


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


def main():
	X, y = gen_data.gen_data(n=2000, is_show=True, random_state=42)
	print(X.shape, collections.Counter(y))
	n, d = X.shape
	out_dim = d
	net = Net(in_dim=2 * d, out_dim=2 * out_dim)
	print(net)
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight
	# create your optimizer
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	criterion = nn.MSELoss(reduction='sum')
	# criterion = nn.CrossEntropyLoss()
	n_epochs = 100
	batch_size = 128
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
			# out = (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(
			# 	axis=1)/out_dim  # squared euclidean distance of X1 and X2
			# out = (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(axis=1)
			d1 = dist2_tensor(out[:, :out_dim], out[:, out_dim:])
			out = d1
			# d2 = compute_d2(X_, d, out, out_dim)
			# out = d1 + d2
			loss = criterion(out, y_)
			# print((d1-y_).abs()[:batch_size])
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

	# # for all pairs
	# X_, y_ = compute_Xy(X)
	# X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	# out = net(X_)
	# # print(out)
	# # d - out_d
	# dist_ =  (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(axis=1).detach().numpy() / out_dim
	# y_ = y_.detach().numpy()
	# print(y_ - dist_)
	# print(dist_.shape, y_.shape)
	# print([(y_[i*n+i], dist_[i*n+i]) for i in range(n)])

	# for testing
	plt.scatter(X[:, 0], X[:, 1], c=y)
	plt.show()
	print(X.shape)

	X_, y_ = compute_Xy2(X[0, :], X)  # X1 and the rest data
	# X_, y_ = compute_Xy2(X)  # X1 and the rest data
	# X_, y_ = X_[:n, :], y_[:n]  # X1 to other points
	X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	out = net(X_)
	# print(X[:20])
	# print(out[:20])
	X1 = out[:, :out_dim].detach().numpy()  # for all X1 projected data.
	for i, (v1, v2) in enumerate(zip(X[:20], out[:20])):
		d1 = y_[i].detach().numpy()
		d2 =  (out[i, :out_dim] - out[i, out_dim]).pow(2).sum().detach().numpy()
		print(v1, v2.detach().numpy(),
		      d1, d2, np.sum(np.square(d1-d2)),
		      y[i])

	plt.scatter(X1[:, 0], X1[:, 1])
	plt.title('the first two dimension')
	plt.show()
	print(X1.shape)
	X2 = out[:, out_dim:].detach().numpy()  # for the rest of projected data.
	print(X2.shape)
	plt.scatter(X2[:, 0], X2[:, 1], c=y)
	plt.title('the last two dimension')
	plt.show()

	X3 = out[:, out_dim:out_dim+2].detach().numpy()  # for the rest of projected data.
	print(X3.shape)
	plt.scatter(X3[:, 0], X3[:, 1], c=y)
	plt.title('the 2-3rd dimension')
	plt.show()


if __name__ == '__main__':
	main()
