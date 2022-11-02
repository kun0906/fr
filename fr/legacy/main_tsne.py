"""Incremental TSNE
"""
import collections
import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
from sklearn.manifold import TSNE

from datasets import gen_data
from inc_tsne import INC_TSNE

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)


def main():
	out_dir = '../out'
	# data_name = '2gaussians'
	data_name = '2circles'
	# data_name = 's-curve'
	# data_name = 'mnist'
	# data_name = '5gaussians-5dims'
	data_name = '3gaussians-10dims'
	is_show = True

	X, y = gen_data.gen_data(n=200, data_type=data_name, is_show=False, random_state=42)
	X, y = sklearn.utils.shuffle(X, y, random_state=42)
	print(X.shape, collections.Counter(y))
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))
	n, d = X.shape

	tsne = TSNE(perplexity=30, method="exact", random_state=42)
	inc_tsne = INC_TSNE(perplexity=30, method="exact", random_state=42)

	n_init = 500
	X_pre = copy.deepcopy(X[:n_init])
	y_pre = y[:n_init]

	res = {'tsne': [], 'inc_tsne': []}

	# 1. Batch TSNE
	st = time.time()
	tsne.fit(X_pre)
	ed = time.time()
	tsne_duration = ed - st
	res['tsne'].append(tsne_duration)
	X_tsne = tsne.embedding_

	# 2. Incremental TSNE
	st = time.time()
	inc_tsne.fit(X_pre)
	ed = time.time()
	inc_tsne_duration = ed - st
	res['inc_tsne'].append(inc_tsne_duration)
	X_inc_tsne = inc_tsne.embedding_


	batch = 0
	if is_show:
		# f = os.path.join(out_dir, f'batch_{batch}.png')
		nrows, ncols = 1, 2
		fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
		ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
		ax[0].set_title(f'TSNE on initial X{X_pre.shape} takes {tsne_duration:.2f}s')
		ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
		ax[1].set_title(f'INC_TSNE on initial X{X_pre.shape} takes {inc_tsne_duration:.2f}s')
		# fig.suptitle(f'INC_TSNE X: batch_{batch}')
		plt.tight_layout()
		# plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()
	# plt.clf()
	# plt.close()
	print(f'Initial X: {X_pre.shape}, tsne:{tsne_duration} vs. inc_tsne: {inc_tsne_duration}.')

	res_file = os.path.join(out_dir, 'res.out')
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

	print(res_file)
	with open(res_file, 'wb') as f:
		pickle.dump(res, f)

	f = os.path.join(out_dir, 'res.png')
	x = range(len(res['tsne']))
	plt.plot(x, res['tsne'], '-ob', label='tsne')
	plt.plot(x, res['inc_tsne'], '-+g', label='inc_tsne')
	plt.xlabel(f'Batch')
	plt.ylabel('Duration (s)')
	plt.title('TSNE vs. INC_TSNE')
	plt.legend()
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()


if __name__ == '__main__':
	main()
