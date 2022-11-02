"""Incremental TSNE
"""
import collections
import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, _utils
from sklearn.manifold._t_sne import _kl_divergence_bh, _kl_divergence, _gradient_descent, MACHINE_EPSILON, \
	_joint_probabilities
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from datasets import gen_data

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)


def _kl_divergence_updated(
		params,
		P,
		degrees_of_freedom,
		n_samples,
		n_components,
		skip_num_points=0,
		compute_error=True,
):
	"""t-SNE objective function: gradient of the KL divergence
	of p_ijs and q_ijs and the absolute error.

	Parameters
	----------
	params : ndarray of shape (n_params,)
		Unraveled embedding.

	P : ndarray of shape (n_samples * (n_samples-1) / 2,)
		Condensed joint probability matrix.

	degrees_of_freedom : int
		Degrees of freedom of the Student's-t distribution.

	n_samples : int
		Number of samples.

	n_components : int
		Dimension of the embedded space.

	skip_num_points : int, default=0
		This does not compute the gradient for points with indices below
		`skip_num_points`. This is useful when computing transforms of new
		data where you'd like to keep the old data fixed.

	compute_error: bool, default=True
		If False, the kl_divergence is not computed and returns NaN.

	Returns
	-------
	kl_divergence : float
		Kullback-Leibler divergence of p_ij and q_ij.

	grad : ndarray of shape (n_params,)
		Unraveled gradient of the Kullback-Leibler divergence with respect to
		the embedding.
	"""
	X_embedded = params.reshape(n_samples, n_components)

	# Q is a heavy-tailed distribution: Student's t-distribution
	dist = pdist(X_embedded, "sqeuclidean")
	dist /= degrees_of_freedom
	dist += 1.0
	dist **= (degrees_of_freedom + 1.0) / -2.0
	Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

	# Optimization trick below: np.dot(x, y) is faster than
	# np.sum(x * y) because it calls BLAS

	# Objective: C (Kullback-Leibler divergence of P and Q)
	if compute_error:
		kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
	else:
		kl_divergence = np.nan

	# Gradient: dC/dY
	# pdist always returns double precision distances. Thus we need to take
	grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
	PQd = squareform((P - Q) * dist)
	for i in range(skip_num_points, n_samples):
		grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
	grad = grad.ravel()
	c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
	grad *= c

	return kl_divergence, grad


class INC_TSNE(TSNE):

	def __init__(
			self,
			n_components=2,
			*,
			perplexity=30.0,
			early_exaggeration=12.0,
			learning_rate="warn",
			n_iter=1000,
			n_iter_without_progress=300,
			min_grad_norm=1e-7,
			metric="euclidean",
			metric_params=None,
			init="warn",
			verbose=0,
			random_state=None,
			method="barnes_hut",
			angle=0.5,
			n_jobs=None,
			square_distances="deprecated",
	):
		# super(self, TSNE)
		self.n_components = n_components
		self.perplexity = perplexity
		self.early_exaggeration = early_exaggeration
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.n_iter_without_progress = n_iter_without_progress
		self.min_grad_norm = min_grad_norm
		self.metric = metric
		self.metric_params = metric_params
		self.init = init
		self.verbose = verbose
		self.random_state = random_state
		self.method = method
		self.angle = angle
		self.n_jobs = n_jobs
		self.square_distances = square_distances

	def _update_X(self, X_embedding_, X_batch):
		random_state = check_random_state(self.random_state)
		n_samples, _ = X_batch.shape

		if self._init == "pca":
			pca = PCA(
				n_components=self.n_components,
				svd_solver="randomized",
				random_state=random_state,
			)
			# TODO: Be careful of this: if increasing the X_batch, pca should also be updated over time
			#  (not implemented yet!)
			X_embedded = pca.fit_transform(X_batch).astype(np.float32, copy=False)
			# TODO: Update in 1.2
			# PCA is rescaled so that PC1 has standard deviation 1e-4 which is
			# the default value for random initialization. See issue #18018.
			warnings.warn(
				"The PCA initialization in TSNE will change to "
				"have the standard deviation of PC1 equal to 1e-4 "
				"in 1.2. This will ensure better convergence.",
				FutureWarning,
			)
		# X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
		elif self._init == "random":
			# The embedding is initialized with iid samples from Gaussians with
			# standard deviation 1e-4.
			X_embedded = 1e-4 * random_state.standard_normal(
				size=(n_samples, self.n_components)
			).astype(np.float32)
		else:
			raise ValueError("'init' must be 'pca', 'random', or a numpy array")

		X_embedded = np.concatenate([X_embedding_, X_embedded], axis=0)

		return X_embedded

	def _update_P(self, P, X, X_batch, desired_perplexity, verbose):

		P = P * self.sum_P  # previous sum_P
		d_up = scipy.spatial.distance.cdist(X, X_batch)
		d_down = d_up.T
		d_diag = scipy.spatial.distance.cdist(X_batch, X_batch)
		conditional_P_up = _utils._binary_search_perplexity(
			d_up, desired_perplexity, verbose
		)
		conditional_P_down = _utils._binary_search_perplexity(
			d_down, desired_perplexity, verbose
		)
		conditional_P_diag = _utils._binary_search_perplexity(
			d_diag, desired_perplexity, verbose
		)

		P_up = np.concatenate([P, conditional_P_up], axis=1)
		P_down = np.concatenate([conditional_P_down, conditional_P_diag], axis=1)
		P = np.concatenate([P_up, P_down], axis=0)

		# assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

		# Normalize the joint probability distribution
		self.sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
		P = P / self.sum_P

		assert np.all(np.abs(P.data) <= 1.0)

		return P

	def update(self, X, X_embedding_, X_batch, P=None, skip_num_points=0):

		X = np.concatenate([X, X_batch], axis=0)
		# # for debugging.
		# random_state = check_random_state(self.random_state)
		# n_samples, _ = X.shape
		# X_embedded = 1e-4 * random_state.standard_normal(
		# 	size=(n_samples, self.n_components)
		# ).astype(np.float32)

		X_embedded = self._update_X(X_embedding_, X_batch)
		n_samples, _ = X_embedded.shape

		# P = self._update_P(P, X, X_batch, desired_perplexity = 5, verbose = 2)
		# compute the joint probability distribution for the input space
		# d_up = scipy.spatial.distance.cdist(X, X_batch)
		# d_down = d_up.T
		# d_diag = scipy.spatial.distance.cdist(X_batch, X_batch)
		distances = pairwise_distances(X, metric=self.metric, squared=True)
		P = _joint_probabilities(distances, self.perplexity, self.verbose)

		# Degrees of freedom of the Student's t-distribution. The suggestion
		# degrees_of_freedom = n_components - 1 comes from
		# "Learning a Parametric Embedding by Preserving Local Structure"
		# Laurens van der Maaten, 2009.
		degrees_of_freedom = max(self.n_components - 1, 1)
		neighbors_nn = None
		self.embedding_ = self._tsne(
			P,
			degrees_of_freedom,
			n_samples,
			X_embedded=X_embedded,
			neighbors=neighbors_nn,
			skip_num_points=skip_num_points,
		)
		return self.embedding_

	def _tsne1(
			self,
			P,
			degrees_of_freedom,
			n_samples,
			X_embedded,
			neighbors=None,
			skip_num_points=0,
	):
		"""Runs t-SNE."""
		# t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
		# and the Student's t-distributions Q. The optimization algorithm that
		# we use is batch gradient descent with two stages:
		# * initial optimization with early exaggeration and momentum at 0.5
		# * final optimization with momentum at 0.8
		params = X_embedded.ravel()

		opt_args = {
			"it": 0,
			"n_iter_check": self._N_ITER_CHECK,
			"min_grad_norm": self.min_grad_norm,
			"learning_rate": self._learning_rate,
			"verbose": self.verbose,
			"kwargs": dict(skip_num_points=skip_num_points),
			"args": [P, degrees_of_freedom, n_samples, self.n_components],
			"n_iter_without_progress": self._EXPLORATION_N_ITER,
			"n_iter": self._EXPLORATION_N_ITER,
			"momentum": 0.5,
		}
		if self.method == "barnes_hut":
			obj_func = _kl_divergence_bh
			opt_args["kwargs"]["angle"] = self.angle
			# Repeat verbose argument for _kl_divergence_bh
			opt_args["kwargs"]["verbose"] = self.verbose
			# Get the number of threads for gradient computation here to
			# avoid recomputing it at each iteration.
			opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
		else:
			obj_func = _kl_divergence

		# Learning schedule (part 1): do 250 iteration with lower momentum but
		# higher learning rate controlled via the early exaggeration parameter
		P *= self.early_exaggeration
		params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
		if self.verbose:
			print(
				"[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
				% (it + 1, kl_divergence)
			)

		# Learning schedule (part 2): disable early exaggeration and finish
		# optimization with a higher momentum at 0.8
		P /= self.early_exaggeration
		remaining = self.n_iter - self._EXPLORATION_N_ITER
		if it < self._EXPLORATION_N_ITER or remaining > 0:
			opt_args["n_iter"] = self.n_iter
			opt_args["it"] = it + 1
			opt_args["momentum"] = 0.8
			opt_args["n_iter_without_progress"] = self.n_iter_without_progress
			params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

		# Save the final number of iterations
		self.n_iter_ = it

		if self.verbose:
			print(
				"[t-SNE] KL divergence after %d iterations: %f"
				% (it + 1, kl_divergence)
			)

		X_embedded = params.reshape(n_samples, self.n_components)
		self.kl_divergence_ = kl_divergence

		return X_embedded


def main():
	out_dir = '../out'
	# data_name = '2gaussians'
	data_name = '2circles'
	# data_name = 's-curve'
	# data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	data_name = '3gaussians-10dims'
	is_show = True
	res = {'tsne': [], 'inc_tsne': []}

	X, y = gen_data.gen_data(n=200, data_type=data_name, is_show=False, random_state=42)
	X, y = sklearn.utils.shuffle(X, y, random_state=42)
	print(X.shape, collections.Counter(y))
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	# Incremental TSNE
	n, d = X.shape
	tsne = TSNE(perplexity=30, method="exact", random_state=42)
	inc_tsne = INC_TSNE(perplexity=30, method="exact", random_state=42)

	n_init = 500
	X_pre = copy.deepcopy(X[:n_init])
	y_pre = y[:n_init]
	# Incremental TSNE
	st = time.time()
	inc_tsne.fit(X_pre)
	ed = time.time()
	inc_tsne_duration = ed - st
	res['inc_tsne'].append(inc_tsne_duration)

	bs = 100
	for i in range(n_init, n, bs):
		print(f'i: {i}/{n}')
		X_batch = X[i:i + bs]
		y_batch = y[i:i + bs]

		# Incremental TSNE
		st = time.time()
		inc_tsne.update(X_pre, inc_tsne.embedding_, X_batch)
		ed = time.time()
		inc_tsne_duration = ed - st
		res['inc_tsne'].append(inc_tsne_duration)
		X_inc_tsne = inc_tsne.embedding_

		# Batch TSNE
		st = time.time()
		# It should be included concatenate operations inside TSNE because inc_tsne contains this step.
		X_pre = np.concatenate([X_pre, X_batch], axis=0)
		y_pre = np.concatenate([y_pre, y_batch], axis=0)
		tsne.fit(X_pre)
		ed = time.time()
		tsne_duration = ed - st
		res['tsne'].append(tsne_duration)
		X_tsne = tsne.embedding_

		batch = (i - n_init) // bs + 1

		if is_show:
			# f = os.path.join(out_dir, f'batch_{batch}.png')
			nrows, ncols = 1, 2
			fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
			ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
			ax[0].set_title(f'TSNE on X{X_pre.shape} takes {tsne_duration:.2f}s')
			ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
			ax[1].set_title(f'INC_TSNE on batch_{batch}{X_batch.shape} takes {inc_tsne_duration:.2f}s')
			# fig.suptitle(f'INC_TSNE X: batch_{batch}')
			plt.tight_layout()
			# plt.savefig(f, dpi=600, bbox_inches='tight')
			plt.show()
		# plt.clf()
		# plt.close()
		print(f'batch_{batch}, tsne:{tsne_duration} vs. inc_tsne: {inc_tsne_duration}.')


if __name__ == '__main__':
	main()
