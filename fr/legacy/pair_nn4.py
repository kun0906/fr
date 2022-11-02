"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import collections
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist, cdist
from torch.optim.lr_scheduler import ExponentialLR

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
		dim = max(in_dim//8, out_dim * 8*2)
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
		# x = F.leaky_relu(self.fc20(x))
		# x = F.sigmoid(self.fc3(x))
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
	X_raw, y_raw = gen_data.gen_data(n=100, data_type='moon1', is_show=False, random_state=42)
	print(X_raw.shape, collections.Counter(y_raw))

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
	model_file = f'net_ep_{n_epochs}-n4.pt'
	if os.path.exists(model_file):
		os.remove(model_file)

	if not os.path.exists(model_file):
		net = Net(in_dim=2 * d, out_dim=2 * out_dim)
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
		batch_size = 100
		history = {'loss': []}
		for epoch in range(n_epochs):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			n, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
			losses = []
			L = 0
			seen = {}
			for i in range(0, n, batch_size):
				X2 = X[i:i + batch_size, :]

				for r in range(X2.shape[0]):  # replace it with the nearest neighbours to fast the training.
					# X_, y_ = compute_Xy2(X2[r], X2)
					X_ = np.asarray([np.concatenate([X2[r], X2[c]]) for c in range(X2.shape[0])])
					y_cos_ = np.asarray(
						[np.asarray(cos_sim(X2[r], X2[c])) for c in range(X2.shape[0])])  # cosine similarity: [-1, 1]
					# X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
					X_ = torch.from_numpy(X_).float()
					y_cos_ = torch.from_numpy(y_cos_).float()
					# in your training loop:

					optimizer.zero_grad()  # zero the gradient buffers
					loss = 0

					out = net(X_)
					# out = (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(
					# 	axis=1)/out_dim  # squared euclidean distance of X1 and X2
					# out = (out[:, :out_dim] - out[:, out_dim:]).pow(2).sum(axis=1)
					# d1 = dist2_tensor(out[:, :out_dim], out[:, out_dim:])
					# loss = criterion(d1, y_)
					# loss = (d1-y_).pow(2).sum()
					# print(r, y_.shape, d1.shape)
					# d2 = torch.cdist(out[:, :out_dim], out[r, out_dim:].reshape((1, out_dim))).sum()  # should be 0
					# loss += d2
					# d2 = torch.cdist(out[:, :out_dim], out[r, :out_dim].reshape((1, out_dim))).sum()  # should be 0
					# loss += d2
					# print(r, X2.shape[0])
					ss1 = np.sum(np.exp(cdist(X2[r, :].reshape((1, -1)), X2)))
					ss2 = torch.cdist(out[r, :out_dim].reshape((1, -1)), out[r, out_dim:].reshape((1, out_dim))).exp().sum()# should be 0

					alpha = 1
					for c in range(X2.shape[0]):
						y_cos2_ = cos_sim_torch(out[r, :out_dim], out[c, out_dim:])
						loss += alpha * (torch.exp((y_cos2_ - y_cos_[c]).abs().sum())-1).abs()
						cos2_ = cos_sim_torch(out[r, :out_dim], out[c, :out_dim])  # itself
						loss += alpha * (torch.exp((cos2_ - 1).abs().sum())-1).abs()

						# d1 = np.square(X2[r]-X2[c])
						# l1 = max(np.linalg.norm(d1), 1e-5)
						# d2 = (out[r, :out_dim] - out[c, out_dim:]).pow(2)
						# l2 = max(torch.norm(d2), 1e-5)
						# l1 = l2 = 1
						# loss += (np.sum(d1)/l1 - d2.sum()/l2).pow(2).sum()

						d1 = np.abs(X2[r] - X2[c])
						d2 = (out[r, :out_dim] - out[c, out_dim:]).abs()
						# loss += (np.max(d1) - torch.max(d2)).pow(2).sum()  # max difference
						# loss += (np.sum(d1) - torch.sum(d2)).abs() # sum difference
						# loss += (out[r, :out_dim] - out[c, :out_dim]).abs().sum()
						# d1 = np.exp(-np.sum(d1))
						# d2 = torch.exp(-d2.sum())
						# loss += d1 * torch.log(d1/d2) + (1-d1)*torch.log((1-d1)/(1-d2)+1e-5)  # cross entropy
						beta = 1
						loss += (torch.exp((np.sum(d1) - beta * torch.sum(d2)).pow(2).sum())-1).abs()
						# (2, 4)/6 == (1, 2)/3, so dividing ss1 (6, 3) only maintain the ratio, not the distance in original space.
						# loss += (torch.exp((np.sum(d1) / ss1 - beta * torch.sum(d2) / ss2).pow(2).sum()) - 1).abs()
						d3 = (out[r, out_dim:] - out[c, :out_dim]).abs()
						loss += (torch.exp(d3.sum())-1).abs()

					# d3 = (out[r, :out_dim]-out[r, out_dim:]).pow(2).sum()   # dist(X1', X1') # the distance of itself.
					# loss += d3
					# print((d1-y_).abs()[:batch_size])

					loss.backward()
					optimizer.step()  # Does the update
					losses.append(loss.item())
					L += loss.item()
			print(f'{epoch + 1}/{n_epochs}, loss: {L}')
			history['loss'].append(L)
		if epoch % 50==0:
			print(scheduler.get_lr())
			scheduler.step()  # adjust learning rate

		plt.plot(history['loss'])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.show()

		# save model
		with open(model_file, 'wb') as f:
			torch.save(net, f)
	else:
		net = torch.load(model_file)

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
	X, y = X_raw, y_raw
	plt.scatter(X[:, 0], X[:, 1], c=y)
	plt.show()
	print(X.shape)

	m = n // n  # n//5
	indices = range(0, n, m)
	X, y = X_raw[indices, :], y_raw[indices]
	plt.figure(figsize=(5, 4))
	plt.scatter(X[:, 0], X[:, 1], c=['g' if v == 0 else 'r' for v in y])
	for i in range(X.shape[0]):
		txt = f'{i}:{X[i]}'
		plt.annotate(txt, (X[i, 0], X[i, 1]))
	plt.title(f'5 points')
	plt.subplots_adjust(right=0.8)
	plt.show()
	print(X.shape)

	nrows, ncols = 6, 7
	fig, ax = plt.subplots(nrows, ncols, figsize=(25, 20))  # width, height
	# fig = plt.figure(figsize=(15, 15))  # width, height
	f = os.path.join('../out', f'projection.png')
	check_path(os.path.dirname(f))
	r = 0
	X_res = []
	y_res = []
	for i in range(X.shape[0]):
		print(f'Showing X_{i} ...')
		X_, y_ = compute_Xy2(X[i, :], X)  # X1 and the rest data
		# X_, y_ = compute_Xy2(X)  # X1 and the rest data
		# X_, y_ = X_[:n, :], y_[:n]  # X1 to other points
		X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
		out = net(X_)
		# print(X[:20])
		# print(out[:20])
		X_res.append(np.mean(out[:, :out_dim].detach().numpy(), axis=0))
		y_res.append(y[i])

		if i >= 3:
			print('break: only show the first few Xs')
			continue
		r = i
		c = 0
		# original space
		ax[r][c].scatter(X[:, 0], X[:, 1], c=y)
		# for i_ in range(X.shape[0]):
		# 	txt = f'{i_}:{X[i_]}'
		# 	ax[r][c].annotate(txt, (X[i_, 0], X[i_, 1]))
		ax[r][c].set_title(f'Original X')
		# plt.show()
		c += 1

		### Projected space
		X1 = np.around(out[:, :out_dim].detach().numpy(), decimals=3)  # for all X1 projected data.
		print(X1.shape, np.max(X1, axis=0) - np.min(X1, axis=0))
		ax[r][c].scatter(X1[:, 0], X1[:, 1])
		ax[r][c].set_title(f'Fixed X_{i}: the first two dimension\n{X[i, :]}->{X1[i, :]}')
		# plt.show()
		c += 1

		X2 = out[:, out_dim:].detach().numpy()  # for the rest of projected data.
		print(X2.shape, np.max(X2, axis=0) - np.min(X2, axis=0))
		ax[r][c].scatter(X2[:, 0], X2[:, 1], c=y)
		# for i_ in range(X2.shape[0]):
		# 	txt = f'{i_}:{X2[i_]}'
		# 	ax[r][c].annotate(txt, (X2[i_, 0], X2[i_, 1]))
		ax[r][c].set_title(f'Fixed X_{i}: the last two dimension')
		# plt.show()
		c += 1

		X3 = out[:, out_dim - 1:out_dim + 1].detach().numpy()  # for the rest of projected data.
		print(X3.shape, np.max(X3, axis=0) - np.min(X3, axis=0))
		ax[r][c].scatter(X3[:, 0], X3[:, 1], c=y)
		ax[r][c].set_title(f'Fixed X_{i}:the 2-3rd dimension')
		c += 1

		res = {'d1': [], 'd2': [], 'diff': [], 'y': [], 'cos1': [], 'cos2': []}
		for i_, (v1, v2) in enumerate(zip(X, out)):
			d1 = y_[i_].detach().numpy()
			d2 = (out[i_, :out_dim] - out[i_, out_dim:]).pow(2).sum().detach().numpy()
			diff = (np.square(d1 - d2))
			cos1 = cos_sim(X[i, :], X[i_, :])
			cos2 = cos_sim_torch(out[i_, :out_dim], out[i_, out_dim:]).detach().numpy()
			if i_ < 20: print(v1, v2.detach().numpy(),
			                  'd1:', d1, 'd2:', d2, 'diff:', diff,
			                  'cos1:', cos1, 'cos2:', cos2,
			                  y[i_])
			res['d1'].append(d1)
			res['d2'].append(d2)
			res['diff'].append(diff)
			res['cos1'].append(cos1)
			res['cos2'].append(cos2)
			res['y'].append(y[i_])

		ax[r][c].plot(res['d1'], 'b*-', label='$d_1: dist(x_i-x_j)$')
		ax[r][c].plot(res['d2'], 'go-', label="$d_2: dist(x_i^{'}-x_j^{'})$")
		ax[r][c].legend()
		ax[r][c].set_title('d1 and d2')
		c += 1

		ax[r][c].plot(res['diff'], 'rv-', label='$diff: dist(d_1-d_2)$')
		ax[r][c].legend()
		ax[r][c].set_title('diff')
		c += 1

		ax[r][c].plot(res['cos1'], 'b*-', label='$cos1: cos(x_i, x_j)$')
		ax[r][c].plot(res['cos2'], 'go-', label="$cos2: cos(x_i^{'}, x_j^{'})$")
		ax[r][c].legend()
		ax[r][c].set_title('cosine similarity')
		c += 1

	r = -1
	c = 0
	X_res = np.asarray(X_res)
	print(X_res.shape)
	ax[r][c].scatter(X_res[:, 0], X_res[:, 1], c=y_res)
	ax[r][c].set_title('final')
	# fig.suptitle(title + fig_name + ', centroids update')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close(fig)


if __name__ == '__main__':
	main()
