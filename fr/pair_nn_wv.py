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
		self.fc1 = nn.Linear(in_dim, 10)
		self.fc2 = nn.Linear(10, 5)
		self.fc20 = nn.Linear(5, 5)
		self.fc3 = nn.Linear(5, out_dim)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		# x = F.tanh(self.fc20(x))
		# x = F.tanh(self.fc20(x))
		# x = F.tanh(self.fc20(x))
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
	return (X1 - X2).pow(2).sum()


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
	return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
	# return np.dot(x1, x2)

def cos_sim_torch(x1, x2):
	return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))


def main():
	X_raw, y_raw = gen_data.gen_data(n=20, is_show=False, random_state=42)
	print(X_raw.shape, collections.Counter(y_raw))
	n, d = X_raw.shape
	out_dim = d
	n_epochs = 50
	model_file = f'net_ep_{n_epochs}-wv.pt'
	# if os.path.exists(model_file):
	# 	os.remove(model_file)

	if not os.path.exists(model_file):
		net = Net(in_dim=3 * d, out_dim=3 * out_dim)
		print(net)
		params = list(net.parameters())
		print(len(params))
		print(params[0].size())  # conv1's .weight
		# create your optimizer
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		criterion = nn.MSELoss(reduction='sum')
		# criterion = nn.CrossEntropyLoss()
		scheduler = ExponentialLR(optimizer, gamma=0.9)
		batch_size = 128
		history = {'loss': []}
		for epoch in range(n_epochs):
			# shuffle X, y
			X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
			n, d = X.shape
			X, y = sklearn.utils.shuffle(X, y, random_state=epoch)
			losses = []
			loss_ = 0
			for i in range(0, n, batch_size):
				X2 = X[i:i + batch_size, :]
				m = X2.shape[0]
				for r in range(m):    # replace it with the nearest neighbours to fast the training.
					for k in range(r, m):
						for t in range(k, m):
							d12 = dist2(X2[r], X2[k])   # d12 == d21
							d13 = dist2(X2[r], X2[t])   # d13 == d31
							d23 = dist2(X2[k], X2[t])   # d23 == d32

							c12 = cos_sim(X2[r], X2[k])
							c13 = cos_sim(X2[r], X2[t])
							c23 = cos_sim(X2[k], X2[t])

							X_ = np.concatenate([X2[r], X2[k], X2[t]])
							y_ = np.asarray([d12, d13, d23])
							y_cos_ = np.asarray([c12, c13, c23])

							X_ = torch.from_numpy(X_).float()
							y_ = torch.from_numpy(y_).float()
							y_cos_ = torch.from_numpy(y_cos_).float()

							# in your training loop:
							optimizer.zero_grad()  # zero the gradient buffers
							O = net(X_)
							d12_O = dist2_tensor(O[:out_dim], O[out_dim:2*out_dim])
							d13_O = dist2_tensor(O[:out_dim], O[2*out_dim:3 * out_dim])
							d23_O = dist2_tensor(O[out_dim:2*out_dim], O[2*out_dim:3 * out_dim])

							cos12_O = cos_sim_torch(O[:out_dim], O[out_dim:2 * out_dim])
							cos13_O = cos_sim_torch(O[:out_dim], O[2 * out_dim:3 * out_dim])
							cos23_O = cos_sim_torch(O[out_dim:2 * out_dim], O[2 * out_dim:3 * out_dim])

							loss = (d12_O-y_[0]).pow(2).sum() + (d13_O-y_[1]).pow(2).sum() + (d23_O-y_[2]).pow(2).sum()
							loss += 10* (cos12_O - y_cos_[0]).pow(2).sum() + (cos13_O - y_cos_[1]).pow(2).sum() + (cos23_O - y_cos_[2]).pow(
								2).sum()

							loss.backward()
							optimizer.step()  # Does the update
							losses.append(loss.item())
							loss_ += loss.item()

			print(f'{epoch + 1}/{n_epochs}, loss: {loss}')
			history['loss'].append(loss.item())
			if epoch % 100: scheduler.step()  # adjust learning rate

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
	f = os.path.join('out', f'projection.png')
	check_path(os.path.dirname(f))
	r = 0
	for i in range(X.shape[0]):
		print(f'Showing X_{i} ...')
		if i >= nrows: break
		# X_, y_ = compute_Xy3(X[i, :], X[i+1, :], X)  # X1 and the rest data
		# # X_, y_ = compute_Xy2(X)  # X1 and the rest data
		# # X_, y_ = X_[:n, :], y_[:n]  # X1 to other points
		out = []
		for j in range(X.shape[0]):
			x1, x2, x3 = X[0], X[1], X[j]
			d12 = dist2(x1, x2)  # d12 == d21
			d13 = dist2(x1, x3)  # d13 == d31
			d23 = dist2(x2, x3)  # d23 == d32
			X_ = np.concatenate([x1, x2, x3])
			y_ = np.asarray([d12, d13, d23])
			X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
			out_ = net(X_)
			out.append(out_.detach().numpy())
		out = np.asarray(out)

		# print(X[:20])
		# print(out[:20])

		r = i
		c = 0
		# # original space
		# ax[r][c].scatter(X[:, 0], X[:, 1], c=y)
		# for i_ in range(X.shape[0]):
		# 	txt = f'{i_}:{X[i_]}'
		# 	ax[r][c].annotate(txt, (X[i_, 0], X[i_, 1]))
		# ax[r][c].set_title(f'Original X')
		# # plt.show()
		# c += 1

		### Projected space
		X1 = np.around(out[:, :out_dim], decimals=3)  # for all X1 projected data.
		print(X1.shape, np.max(X1, axis=0) - np.min(X1, axis=0))
		ax[r][c].scatter(X1[:, 0], X1[:, 1])
		ax[r][c].set_title(f'Fixed X_{i}: the first two dimension\n{X[i, :]}->{X1[i, :]}')
		# plt.show()
		c += 1

		X2 = out[:, 2*out_dim:]  # for the rest of projected data.
		print(X2.shape, np.max(X2, axis=0) - np.min(X2, axis=0))
		ax[r][c].scatter(X2[:, 0], X2[:, 1], c=y)
		# for i_ in range(X2.shape[0]):
		# 	txt = f'{i_}:{X2[i_]}'
		# 	ax[r][c].annotate(txt, (X2[i_, 0], X2[i_, 1]))
		ax[r][c].set_title(f'Fixed X_{i}: the last two dimension')
		# plt.show()
		c += 1

		# X3 = out[:, out_dim - 1:out_dim + 1].detach().numpy()  # for the rest of projected data.
		# print(X3.shape, np.max(X3, axis=0) - np.min(X3, axis=0))
		# ax[r][c].scatter(X3[:, 0], X3[:, 1], c=y)
		# ax[r][c].set_title(f'Fixed X_{i}:the 2-3rd dimension')
		# c += 1
		#
		res = {'d1': [], 'd2': [], 'diff': [], 'y': [], 'cos1': [], 'cos2': []}
		for i_, (v1, v2) in enumerate(zip(X, out)):
			# d1 = y_[i_].detach().numpy()
			# d2 = (out[i_, :out_dim] - out[i_, out_dim:]).pow(2).sum().detach().numpy()
			# diff = (np.square(d1 - d2))
			d1 =d2=diff = 0
			cos12 = cos_sim(X[0, :], X[1, :])
			cos13 = cos_sim(X[0, :], X[i_, :])
			cos23 = cos_sim(X[1, :], X[i_, :])
			# cos2 = cos_sim_torch(out[i_, :out_dim], out[i_, out_dim:]).detach().numpy()
			cos12_O = cos_sim(out[0, :out_dim], out[1, out_dim:2*out_dim])
			cos13_O = cos_sim(out[0, :out_dim], out[i_, 2*out_dim:])
			cos23_O = cos_sim(out[1, out_dim:2*out_dim], out[i_, 2*out_dim:])
			if i_ < 20: print(v1, v2,
			                  'd1:', d1, 'd2:', d2, 'diff:', diff,
			                  # 'cos1:', cos1, 'cos2:', cos2,
			                  cos12, cos12_O, cos13, cos13_O, cos23, cos23_O,
			                  y[i_])
			res['d1'].append(d1)
			res['d2'].append(d2)
			res['diff'].append(diff)
			# res['cos1'].append(cos1)
			# res['cos2'].append(cos2)
			res['y'].append(y[i_])

		# ax[r][c].plot(res['d1'], 'b*-', label='$d_1: dist(x_i-x_j)$')
		# ax[r][c].plot(res['d2'], 'go-', label="$d_2: dist(x_i^{'}-x_j^{'})$")
		# ax[r][c].legend()
		# ax[r][c].set_title('d1 and d2')
		# c += 1

		# ax[r][c].plot(res['diff'], 'rv-', label='$diff: dist(d_1-d_2)$')
		# ax[r][c].legend()
		# ax[r][c].set_title('diff')
		# c += 1
		#
		# ax[r][c].plot(res['cos1'], 'b*-', label='$cos1: cos(x_i, x_j)$')
		# ax[r][c].plot(res['cos2'], 'go-', label="$cos2: cos(x_i^{'}, x_j^{'})$")
		# ax[r][c].legend()
		# ax[r][c].set_title('cosine similarity')
		# c += 1

	# fig.suptitle(title + fig_name + ', centroids update')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	# plt.clf()
	plt.close(fig)


if __name__ == '__main__':
	main()
