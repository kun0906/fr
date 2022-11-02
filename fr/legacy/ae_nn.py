"""

    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import collections

import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import gen_data
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        hidden_dim = 10
        self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.Tanh(),
                nn.Linear(hidden_dim//2, out_dim)
        )

        self.decoder = nn.Sequential(
                nn.Linear(out_dim, hidden_dim//2),
                nn.Tanh(),
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, in_dim),
                # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#
# class Net(nn.Module):
#
# 	def __init__(self, in_dim, out_dim=2):
# 		super(Net, self).__init__()
# 		# an affine operation: y = Wx + b
# 		self.fc1 = nn.Linear(in_dim, 10)  # 5*5 from image dimension
# 		self.fc2 = nn.Linear(10, 5)
# 		self.fc3 = nn.Linear(5, 2 * out_dim)
#
# 	def forward(self, x):
# 		x = F.sigmoid(self.fc1(x))
# 		x = F.sigmoid(self.fc2(x))
# 		x = self.fc3(x)
# 		return x
#

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
		y_.append(dist2(X1, X[i]) / d)

	return np.asarray(X_), np.asarray(y_)

def dist2_tensor(X1, X2):
	return (X1-X2).pow(2).sum(axis=1)

def main():
	X, y = gen_data.gen_data(n=100, data_type='mnist1', is_show=True, random_state=42)
	X = X/255
	print(X.shape, collections.Counter(y))
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))
	n, d = X.shape
	in_dim = d
	out_dim = 2
	net = AutoEncoder(in_dim=2 * in_dim, out_dim=2* out_dim)
	print(net)
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight
	# create your optimizer
	optimizer = optim.Adam(net.parameters(), lr=0.0001)
	criterion = nn.MSELoss()
	# criterion = nn.CrossEntropyLoss()
	n_epochs = 200
	batch_size = 32
	history = {'loss': []}
	lambda_ = 0.1
	lambda2_ = 0.1
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
			encoded, decoded = net(X_)
			d1 = dist2_tensor(encoded[:, :out_dim], encoded[:, out_dim:])/out_dim # z1 -z2
			d2 = dist2_tensor(X_[:, :in_dim], X_[:, in_dim:]) / in_dim  # x1 -x2
			d3 = dist2_tensor(X_[:, :in_dim], decoded[:, :in_dim])/in_dim + \
			      dist2_tensor(X_[:, in_dim:], decoded[:, in_dim:]) / in_dim
			out = lambda_ * ((lambda2_*d1 - (1-lambda2_)*d2).pow(2)) + (1-lambda_) * d3
				# squared euclidean distance of X1 and X2
			# loss = criterion(out, torch.zeros(y_.shape))
			loss = torch.mean(out)
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

	# for all pairs for debugging
	X_, y_ = compute_Xy(X)
	X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	encoded, decoded = net(X_)
	# print(out)
	# d - out_d
	dist_ =  dist2_tensor(encoded[:, :out_dim], encoded[:, out_dim:]).detach().numpy() / out_dim
	y_ = y_.detach().numpy()
	print(y_ - dist_)
	print(dist_.shape, y_.shape)
	print([(y_[i*n+i], dist_[i*n+i]) for i in range(n)][:10])

	dist_1 = dist2_tensor(X_[:, :in_dim], decoded[:, :in_dim]).detach().numpy() / out_dim
	print(dist_1[:10])
	dist_2 = dist2_tensor(X_[:, in_dim:], decoded[:, in_dim:]).detach().numpy() / out_dim
	print(dist_2[:10])


	# for testing
	y2 = y[:n]  # exclude the first point
	X_, y_ = compute_Xy2(X[0, :], X[:, :])  # X1 and the rest data
	# X_, y_ = compute_Xy2(X)  # X1 and the rest data
	X_, y_ = X_[:n, :], y_[:n]  # X1 to other points
	X_, y_ = torch.from_numpy(X_).float(), torch.from_numpy(y_).float()
	encoded, decoded = net(X_)
	X2 = encoded[:, out_dim:].detach().numpy()
	print(X2.shape)
	# y2 =
	# plt.scatter(X2, y2)
	plt.scatter(X2[:, 0], X2[:, 1], c=y2)
	plt.title('Embedded X')
	plt.show()

	X2 = decoded[:, in_dim:].detach().numpy()
	print(X2.shape)
	# y2 =
	# plt.scatter(X2, y2)
	plt.scatter(X2[:, 0], X2[:, 1], c=y2)
	plt.title('Reconstructed X')
	plt.show()


if __name__ == '__main__':
	main()
