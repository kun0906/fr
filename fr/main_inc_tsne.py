"""Incremental TSNE
"""
# Author: kun.bj@outlook.com

import collections
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
from sklearn.neighbors import NearestNeighbors

from datasets import gen_data
from inc_tsne import INC_TSNE
from tsne import TSNE, trustworthiness
from utils.common import check_path, timer, fmt

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)
warnings.simplefilter(action='ignore', category=FutureWarning)


@timer
def animate(figs, out_dir='.'):
	print(figs)
	import imageio
	images = [imageio.v2.imread(f) for f in figs]
	out_file = os.path.join(out_dir, 'batches.gif')
	imageio.mimsave(out_file, images, duration=1)  # each image 0.5s

	return out_file


def evaluate(X, y=None, X_embedded=None):
	res = {}

	# Get trustworthiness
	score = trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean')
	res['trustworthiness'] = score

	# Get 1-nearest neighbor accuracy
	nn = NearestNeighbors(n_neighbors=1)
	nn.fit(X_embedded)
	indices = nn.kneighbors(return_distance=False)  # exclude the query node itself
	y_pred = y[indices]
	acc = sklearn.metrics.accuracy_score(y, y_pred)
	res['acc_1nn'] = acc

	return res


@timer
def main(args):
	is_show = True
	random_state = 42
	res = {'tsne': [], 'inc_tsne': []}
	data_name = args['data_name']
	"""1. Generate dataset 
		
	"""
	X, y = gen_data.gen_data(n=args['n'], data_type=args['data_name'], is_show=False, random_state=random_state)
	X, y = sklearn.utils.shuffle(X, y, random_state=random_state)
	print(f'X.shape: {X.shape}, y: {collections.Counter(y)}')
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	n, d = X.shape
	n_iter = 1000
	init_percent = 0.3
	n_init = int(np.ceil(n * init_percent))
	n_init_iter = int(np.ceil(n_iter * init_percent))
	n1 = int(np.ceil(n_init_iter * (1 / 4)))  # 250:750 = 1:3
	args['init_iters'] = (n1, n_init_iter - n1)
	bs = int(np.ceil((n - n_init) * 0.2))  # bs is 10% of (n-n_init)
	each_iter = int((n_iter - n_init_iter) * 0.2)
	n1 = int(np.ceil(each_iter*(1/4))) # 250:750 = 1:3
	args['update_iters'] = (n1, each_iter-n1)  # when i%5 != 0, learning stage 1 and 2
	sub_dir = '|'.join([str(args['init_iters']), str(args['update_iters'])])
	out_dir = os.path.join(args['out_dir'], sub_dir)
	print(f'n: {n}, n_init: {n_init}, bs: {bs}, args: {args}')

	"""2. Get the initial fitting results for TSNE and INC_TSNE
	"""
	# Incremental TSNE
	Y_update_init = args['update_init']
	tsne = TSNE(perplexity=args['perplexity'], method=args['method'], n_iter=n_iter,
	            random_state=random_state, verbose=0)
	inc_tsne = INC_TSNE(perplexity=args['perplexity'], method=args['method'], update_init=args['update_init'],
						n_iter=n_iter, init_iters = args['init_iters'], update_iters = args['update_iters'],
	                    random_state=random_state, is_last_batch=False, verbose=0)

	X_pre, X, y_pre, y = sklearn.model_selection.train_test_split(X, y, train_size=n_init,
	                                                              random_state=random_state, shuffle=True)
	# Incremental TSNE
	st = time.time()
	inc_tsne.fit(X_pre)
	ed = time.time()
	inc_tsne_duration = ed - st
	inc_tsne.n_iter_ += 1  # start for 0
	print(f'inc_tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(inc_tsne_duration)}s')
	scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
	res['inc_tsne'].append((inc_tsne_duration, inc_tsne.n_iter_, scores))
	X_inc_tsne = inc_tsne.embedding_
	# TSNE
	st = time.time()
	tsne.fit(X_pre)
	ed = time.time()
	tsne_duration = ed - st
	tsne.n_iter_ += 1  # start for 0
	print(f'tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(tsne_duration)}s')
	scores = evaluate(X_pre, y_pre, tsne.embedding_)
	res['tsne'].append((tsne_duration, tsne.n_iter_, scores))
	X_tsne = tsne.embedding_

	if is_show:
		f = os.path.join(out_dir, 'batch', f'init.png')
		nrows, ncols = 1, 2
		fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
		ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
		tsne_duration, tsne_iters, scores = res['tsne'][-1]
		trust = scores['trustworthiness']
		acc = scores['acc_1nn']
		ax[0].set_title(f'TSNE on {data_name}{X_pre.shape} takes {fmt(tsne_duration)}s,\n'
		                f'{tsne_iters} iterations\n'
		                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
		ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
		inc_tsne_duration, inc_tsne_iters, scores = res['inc_tsne'][-1]
		trust = scores['trustworthiness']
		acc = scores['acc_1nn']
		ax[1].set_title(f'INC_TSNE({Y_update_init}) on initial {X_pre.shape} takes {fmt(inc_tsne_duration)}s,\n'
		                f'{inc_tsne_iters} iterations\n'
		                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
		# fig.suptitle(f'INC_TSNE X: batch_{batch}')
		plt.tight_layout()
		check_path(os.path.dirname(f))
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()


	"""3. Get the results for each batch
	"""
	res_file = os.path.join(out_dir, 'res.out')
	i = 1
	batch_figs = []
	while X.shape[0] > 0:
		if bs < X.shape[0]:
			X_batch, X, y_batch, y = sklearn.model_selection.train_test_split(X, y, train_size=bs,
			                                                                  random_state=random_state, shuffle=True)
		else:
			X_batch, X, y_batch, y = X, np.zeros((0,)), y, np.zeros((0,))
		# if i % 5 == 0:
		# 	n_iter = args['update2_iters']
		# else:
		# 	n_iter = args['update1_iters']
		n_iter = args['update_iters']
		print(f'{i}-th batch, X_batch: {X_batch.shape}, y_batch: {collections.Counter(y_batch)}, update_iters: {n_iter}')

		# 3.1 Incremental TSNE
		st = time.time()
		inc_tsne.update(X_pre, X_batch, n_iter)
		ed = time.time()
		inc_tsne_duration = res['inc_tsne'][-1][0] + (ed - st)
		inc_tsne.n_iter_ += 1  # start for 0
		X_pre = np.concatenate([X_pre, X_batch], axis=0)
		y_pre = np.concatenate([y_pre, y_batch], axis=0)
		scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
		res['inc_tsne'].append((inc_tsne_duration, inc_tsne.n_iter_, scores))
		X_inc_tsne = inc_tsne.embedding_

		# 3.2 Batch TSNE
		st = time.time()
		tsne.fit(X_pre)
		ed = time.time()
		tsne_duration = ed - st
		scores = evaluate(X_pre, y_pre, tsne.embedding_)
		tsne.n_iter_ += 1  # start for 0
		res['tsne'].append((tsne_duration, tsne.n_iter_, scores))
		X_tsne = tsne.embedding_

		if is_show:
			f = os.path.join(out_dir, 'batch', f'batch_{i}.png')
			nrows, ncols = 1, 2
			fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
			ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
			tsne_duration, tsne_iters, scores = res['tsne'][-1]
			trust = scores['trustworthiness']
			acc = scores['acc_1nn']
			ax[0].set_title(f'TSNE on {data_name}{X_pre.shape} takes {fmt(tsne_duration)}s,\n'
			                f'{tsne_iters} iterations\n'
			                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
			ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
			inc_tsne_duration, inc_tsne_iters, scores = res['inc_tsne'][-1]
			trust = scores['trustworthiness']
			acc = scores['acc_1nn']
			ax[1].set_title(f'INC_TSNE({Y_update_init}) on batch_{i}{X_batch.shape} takes {fmt(inc_tsne_duration)}s,\n'
			                f'{inc_tsne_iters} iterations\n'
			                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
			# fig.suptitle(f'INC_TSNE X: batch_{batch}')
			plt.tight_layout()
			check_path(os.path.dirname(f))
			plt.savefig(f, dpi=600, bbox_inches='tight')
			plt.show()
			batch_figs.append(f)
		# plt.clf()
		# plt.close()
		inc_tsne_duration_conponents = '+'.join(fmt([res['inc_tsne'][-2][0]]+list(inc_tsne.update_res['time'])))
		tsne_duration_conponents = '+'.join(fmt([0]+list(tsne.fit_res['time'])))
		inc_tsne_kl = ' - '.join([','.join(fmt(list(v))) for v in inc_tsne.update_res['kl_divergence']])
		tsne_kl = ' - '.join([','.join(fmt(list(v))) for v in tsne.fit_res['kl_divergence']])
		print(f'batch_{i},  tsne({tsne_iters}): {fmt(tsne_duration)}s ({tsne_duration_conponents}, kl: {tsne_kl}) '
		      f'vs. \n\t'
		      f'inc_tsne({inc_tsne_iters}): {fmt(inc_tsne_duration)}s ({inc_tsne_duration_conponents}, '
		      f'kl: {inc_tsne_kl}).')
		i += 1

	"""4. Save and plot results
	"""
	# Save results to disk
	print(res_file)
	with open(res_file, 'wb') as f:
		pickle.dump(res, f)

	# Plot final results
	for key in ['duration', 'iteration', 'trustworthiness', 'acc_1nn']:
		x = range(len(res['tsne'][1:]))
		if key == 'duration':
			y = [dur for dur, iters, scores in res['tsne']][1:]
			y_label = 'Duration (s)'
		elif key == 'iteration':
			y = [iters for dur, iters, scores in res['tsne']][1:]
			y_label = 'Iteration'
		else:
			y = [scores[key] for dur, iters, scores in res['tsne']][1:]
			y_label = 'Trustworthiness' if key == 'trustworthiness' else 'ACC_1NN'
		plt.plot(x, y, '-ob', label='tsne')
		if key == 'duration':
			y = [dur for dur, iters, scores in res['inc_tsne']][1:]
		elif key == 'iteration':
			y = [iters for dur, iters, scores in res['inc_tsne']][1:]
			y_label = 'Iteration'
		else:
			y = [scores[key] for dur, iters, scores in res['inc_tsne']][1:]
		plt.plot(x, y, '-+g', label='inc_tsne')
		plt.xlabel(f'Batch')
		plt.ylabel(y_label)
		plt.title(f'TSNE vs. INC_TSNE ({data_name}): {y_label}')
		plt.legend()
		plt.tight_layout()
		f = os.path.join(out_dir, f'TSNE_vs_INCTSNE-{y_label}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

	# Make animation with all batches
	out_file = animate(batch_figs, out_dir)
	print(out_file)


if __name__ == '__main__':
	# data_name = '2gaussians'
	# data_name = '2circles'
	# data_name = 's-curve'
	# data_name = '5gaussians-5dims'
	# data_name = '3gaussians-10dims'
	n = 100
	args_lst = []
	"""Case 0: 2circles
		It includes 2 clusters, and each has 500 data points in R^2.
	"""
	args = {'data_name': '2circles', 'n': n}
	args_lst.append(args)
	"""Case 1: 3gaussians-10dims
		It includes 3 clusters, and each has 500 data points in R^10.
	"""
	args = {'data_name': '3gaussians-10dims', 'n': n}
	args_lst.append(args)
	"""Case 2: mnist
		It includes 10 clusters, and each has 500 data points in R^784.
	"""
	args = {'data_name': 'mnist', 'n': n}
	args_lst.append(args)
	for args in args_lst:
		for update_init in ['Gaussian', 'weighted']:
			perplexity = 30
			args['method'] = "exact"   # 'exact', "barnes_hut"
			args['perplexity'] = perplexity
			args['update_init'] = update_init
			sub_dir = '|'.join([args['method'], f'{perplexity}', args['update_init']])
			args['out_dir'] = os.path.join('out', args['data_name'], str(args['n']), sub_dir)
			print(f'\n\nargs: {args}')
			main(args)
