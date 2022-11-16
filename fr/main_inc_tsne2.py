"""Incremental TSNE
"""
# Author: kun.bj@outlook.com

import collections
import os
import pickle
import shutil
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils

from _base import evaluate, trustworthiness
from datasets import gen_data
from inc_tsne import INC_TSNE
from tsne import TSNE
from utils.common import check_path, timer, fmt

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_colors(y, name='rainbow'):
	"""
		https://stackoverflow.com/questions/52108558/how-does-parameters-c-and-cmap-behave-in-a-matplotlib-scatter-plot
		https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib

	Parameters
	----------
	y
	name

	Returns
	-------

	"""
	cmap = plt.get_cmap(name)
	norm = matplotlib.colors.Normalize(vmin=min(y), vmax=max(y))
	colors = cmap(norm(y), bytes=False)  # RGBA
	return colors


@timer
def animate(figs, out_file='all.mp4'):
	""" MacOS cannot stop the gif loop, so you can view it in Browser (e.g., Chrome).

	Parameters
	----------
	figs
	out_file

	Returns
	-------

	"""
	print(figs)
	import imageio, PIL
	# figs = [imageio.v2.imread(f) for f in figs]
	# kwargs = {'duration': 1, 'loop': 1}
	# imageio.mimsave(out_file, images, format='GIF', **kwargs)  # each image 0.5s = duration/n_imgs
	images = []
	for i, f in enumerate(figs):
		if 'png' not in f: continue
		im = imageio.v2.imread(f)  # RGBA
		if i == 0:
			shape = im.shape[:2][::-1]
		# print(im.shape)
		im = PIL.Image.fromarray(im).resize(shape)  # (width, height)
		images.append(im)
	kwargs = {'fps': 1}
	imageio.v2.mimsave(out_file, images, format='mp4', **kwargs)  # each image 0.5s = duration/n_imgs

	return out_file


def show_embedded_data(tsne, inc_tsne, res, X_pre, y_pre, args, out_dir, idx=0):
	f = os.path.join(out_dir, 'batch', f'batch_{idx}.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), dpi=600)  # width, height
	print(f'figure number: {plt.gcf().number}, {fig.number}')
	data_name = args['data_name']
	Y_update_init = args['update_init']
	# tsne
	X_tsne = tsne.embedding_
	ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
	tsne_duration, tsne_iters, scores = res['tsne'][-1]
	trust = scores['trustworthiness']
	acc = scores['acc_1nn']
	ax[0].set_title(f'TSNE on {data_name}{X_pre.shape} takes {fmt(tsne_duration)}s,\n'
	                f'{tsne_iters} iterations\n'
	                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')

	# inc_tsne
	X_inc_tsne = inc_tsne.embedding_
	ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
	inc_tsne_duration, inc_tsne_iters, scores = res['inc_tsne'][-1]
	trust = scores['trustworthiness']
	acc = scores['acc_1nn']
	trust_diff = res['trust_diff'][0]
	ax[1].set_title(f'INC_TSNE({Y_update_init}) on initial {X_pre.shape} takes {fmt(inc_tsne_duration)}s,\n'
	                f'{inc_tsne_iters} iterations, trust_diff: {fmt(trust_diff)}\n'
	                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
	# fig.suptitle(f'INC_TSNE X: batch_{batch}')
	plt.tight_layout()
	check_path(os.path.dirname(f))
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	plt.close()


def _show_embedded_data(tsne, inc_tsne, X_pre, y_pre, args, out_dir, idx=0, learning_phase=0, jth_iter=0):
	f = os.path.join(out_dir, f'batch/{idx}', f'{learning_phase}-{jth_iter}.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), dpi=600)  # width, height
	if jth_iter % 10 == 0: print(f'{jth_iter}th iteration, figure number: {plt.gcf().number}, {fig.number}')
	data_name = args['data_name']
	Y_update_init = args['update_init']
	n1 = len(tsne._update_data[learning_phase])
	if jth_iter < n1:
		# tsne
		Y = tsne._update_data[learning_phase][jth_iter]['Y']
		scores = tsne._update_data[learning_phase][jth_iter]['scores']
		ax[0].scatter(Y[:, 0], Y[:, 1], c=y_pre)
		trust = scores['trustworthiness']
		acc = scores['acc_1nn']
		ax[0].set_title(f'TSNE on {data_name}{X_pre.shape},\n'
		                f'{learning_phase}th learning_phase ({n1}), {jth_iter}th iteration\n'
		                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')

	n2 = len(inc_tsne._update_data[learning_phase])
	if jth_iter < n2:
		# inc_tsne
		Y = inc_tsne._update_data[learning_phase][jth_iter]['Y']
		scores = inc_tsne._update_data[learning_phase][jth_iter]['scores']
		ax[1].scatter(Y[:, 0], Y[:, 1], c=y_pre)
		trust = scores['trustworthiness']
		acc = scores['acc_1nn']
		ax[1].set_title(f'INC_TSNE({Y_update_init}) on initial {X_pre.shape},\n'
		                f'{learning_phase}th learning_phase ({n2}), {jth_iter}th iteration\n'
		                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
	# fig.suptitle(f'INC_TSNE X: batch_{batch}')
	plt.tight_layout()
	check_path(os.path.dirname(f))
	plt.savefig(f, dpi=600, bbox_inches='tight')
	if jth_iter == 0: plt.show()
	plt.close()

	return f


def _show_update_data(tsne, inc_tsne, out_dir, idx=0):
	f = os.path.join(out_dir, 'batch', f'{idx}.png')
	nrows, ncols = 1, 2
	fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), dpi=600)  # width, height
	print(f'figure number: {plt.gcf().number}, {fig.number}')
	# tsne
	res1, res2 = tsne._update_data
	n1, n2 = len(res1), len(res2)
	trust1 = []
	acc_1nn1 = []
	for _i in range(len(res1)):
		trust1.append(res1[_i]['scores']['trustworthiness'])
		acc_1nn1.append(res1[_i]['scores']['acc_1nn'])
	trust2 = []
	acc_1nn2 = []
	for _i in range(len(res2)):
		trust2.append(res2[_i]['scores']['trustworthiness'])
		acc_1nn2.append(res2[_i]['scores']['acc_1nn'])
	_X = range(n1 + n2)
	# trustworthiness
	trust = trust1 + trust2
	ax[0].plot(_X, trust, color='gray', ls='--', marker='*',
	           markerfacecolor='blue', markeredgecolor='blue', label='tsne')
	ax[0].axvline(x=n1, ls='-', color='black')
	position_trust = float(fmt((max(trust) + min(trust)) / 2))
	ax[0].text(n1 + 0.1, position_trust, f'tsne (n1:{n1})', rotation=90)  # (x, y, text)
	# acc_1nn
	acc = acc_1nn1 + acc_1nn2
	ax[1].plot(_X, acc, color='gray', ls='--', marker='*',
	           markerfacecolor='blue', markeredgecolor='blue', label='tsne')
	ax[1].axvline(x=n1, ls='-', color='black')
	position_acc = float(fmt((max(acc) + min(acc)) / 2))
	ax[1].text(n1 + 0.1, position_acc, f'tsne (n1:{n1})', rotation=90)  # (x, y, text)

	# inc_tsne
	res1, res2 = inc_tsne._update_data
	n1, n2 = len(res1), len(res2)
	trust1 = []
	acc_1nn1 = []
	for _i in range(len(res1)):
		trust1.append(res1[_i]['scores']['trustworthiness'])
		acc_1nn1.append(res1[_i]['scores']['acc_1nn'])
	trust2 = []
	acc_1nn2 = []
	for _i in range(len(res2)):
		trust2.append(res2[_i]['scores']['trustworthiness'])
		acc_1nn2.append(res2[_i]['scores']['acc_1nn'])
	_X = range(n1 + n2)
	# trustworthiness
	trust = trust1 + trust2
	ax[0].plot(_X, trust, color='gray', ls='--', marker='o',
	           markerfacecolor='green', markeredgecolor='green', label='inc_tsne')
	ax[0].axvline(x=n1, ls='-', color='black')
	# position = float(fmt((max(trust) + min(trust)) / 2))
	ax[0].text(n1 + 0.1, position_trust, f'inc_tsne n1:{n1}', rotation=90)  # (x, y, text)

	# acc_1nn
	acc = acc_1nn1 + acc_1nn2
	ax[1].plot(_X, acc, color='gray', ls='--', marker='o',
	           markerfacecolor='green', markeredgecolor='green', label='inc_tsne')
	ax[1].axvline(x=n1, ls='-', color='black')
	# position = float(fmt((max(acc) + min(acc)) / 2))
	ax[1].text(n1 + 0.1, position_acc, f'inc_tsne n1:{n1}', rotation=90)  # (x, y, text)

	ax[0].set_xlabel(f'Iteration')
	y_label = 'Trustworthiness'
	ax[0].set_ylabel(y_label)
	ax[0].legend()
	ax[0].set_title(f'Trustworthiness. Idx: {idx}')
	ax[1].set_xlabel(f'Iteration')
	y_label = 'ACC_1NN'
	ax[1].set_ylabel(y_label)
	ax[1].legend()
	ax[1].set_title(f'ACC_1NN. Idx: {idx}')

	# fig.suptitle(f'INC_TSNE X: batch_{batch}')
	plt.tight_layout()
	check_path(os.path.dirname(f))
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()
	plt.close()


@timer
def main(args):
	is_show = True
	random_state = 42
	res = {'tsne': [], 'inc_tsne': []}
	data_name = args['data_name']
	"""1. Generate dataset 
		
	"""
	X, y = gen_data.gen_data(n=args['n'], data_type=args['data_name'], is_show=False,
	                         with_noise=False, random_state=random_state)
	X, y = sklearn.utils.shuffle(X, y, random_state=random_state)
	print(f'X.shape: {X.shape}, y: {collections.Counter(y)}')
	print(np.quantile(X, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	if args['data_name'] == 'mnist':
		indices = np.where((y % 2 == 0) & (y > 0))[0]
		X = X[indices]
		y = y[indices]

	n, d = X.shape
	n_iter = args['n_iter']
	init_percent = 0.9
	n_init = int(np.round(n * init_percent))  # the total number of initial training data size.
	n_init_iter = int(np.round(n_iter * 1.0))  # the total number of initial iterations
	n1 = int(np.round(n_init_iter * (1 / 4)))  # 250:750 = 1:3
	args['init_iters'] = (n1, n_init_iter - n1)

	batch_percent = 1.0
	bs = int(np.round((n - n_init) * batch_percent))  # bs is 10% of (n-n_init)
	each_iter = int(0.5 * n_iter)  # int((n_iter - n_init_iter) * batch_percent)  # floor
	n1 = int(np.round(each_iter * (1 / 4)))  # 250:750 = 1:3
	print(f'n1: {np.round(each_iter * (1 / 4))}')
	if n1 <= 0 or each_iter - n1 <= 0:
		raise ValueError(f'n_iteration {n_iter} is too small!')
	args['update_iters'] = (n1, each_iter - n1)  # when i%5 != 0, learning stage 1 and 2
	sub_dir = '|'.join([str(args['init_iters']), str(args['update_iters'])])
	out_dir = os.path.join(args['out_dir'], sub_dir)
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	print(f'n: {n}, n_init: {n_init}, bs: {bs}, args: {args}')

	"""2. Get the initial fitting results for TSNE and INC_TSNE
	"""
	# Incremental TSNE
	Y_update_init = args['update_init']
	tsne = TSNE(perplexity=args['perplexity'], method=args['method'], n_iter=n_iter,
	            _EXPLORATION_N_ITER=int(np.round(n_init_iter * (1 / 4))),
	            random_state=random_state, verbose=0)
	inc_tsne = INC_TSNE(perplexity=args['perplexity'], method=args['method'], update_init=args['update_init'],
	                    n_iter=n_iter, init_iters=args['init_iters'], update_iters=args['update_iters'],
	                    random_state=random_state, is_last_batch=False, verbose=0)

	X_pre, X, y_pre, y = sklearn.model_selection.train_test_split(X, y, train_size=n_init, stratify=y,
	                                                              random_state=random_state, shuffle=True)
	# Incremental TSNE
	st = time.time()
	inc_tsne.fit(X_pre, y_pre)
	ed = time.time()
	inc_tsne_duration = ed - st
	inc_tsne.n_iter_ += 1  # start for 0
	print(f'inc_tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(inc_tsne_duration)}s')
	scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
	res['inc_tsne'].append((inc_tsne_duration, inc_tsne.n_iter_, scores))
	X_inc_tsne = inc_tsne.embedding_
	# TSNE
	st = time.time()
	tsne.fit(X_pre, y_pre)
	ed = time.time()
	tsne_duration = ed - st
	tsne.n_iter_ += 1  # start for 0
	print(f'tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(tsne_duration)}s')
	scores = evaluate(X_pre, y_pre, tsne.embedding_)
	res['tsne'].append((tsne_duration, tsne.n_iter_, scores))
	X_tsne = tsne.embedding_

	trust_diff = trustworthiness(X_tsne, X_inc_tsne, n_neighbors=5, metric='euclidean')
	res['trust_diff'] = [trust_diff]
	if is_show:
		show_embedded_data(tsne, inc_tsne, res, X_pre, y_pre, args, out_dir, idx=0)

		# save update data
		_show_update_data(tsne, inc_tsne, out_dir, idx=0)

		show_detail_flg = False
		if show_detail_flg:
			# show each update iteration
			idx_batch = 0
			figs = []
			for learning_phase in [0, 1]:  # two phases of gradient updates with/without early_exaggeration
				# _n = min(5, len(tsne._update_data[learning_phase]), len(inc_tsne._update_data[learning_phase]))
				_n = max(10, len(tsne._update_data[learning_phase]), len(inc_tsne._update_data[learning_phase]))
				for j in range(_n):
					fig = _show_embedded_data(tsne, inc_tsne, X_pre, y_pre, args, out_dir, idx=idx_batch,
					                          learning_phase=learning_phase, jth_iter=j)
					figs.append(fig)
			animate(figs, out_file=os.path.join(out_dir, 'batch', f'{idx_batch}.mp4'))

	"""3. Get the results for each batch
	"""
	res_file = os.path.join(out_dir, 'res.out')
	i = 1
	batch_figs = []
	_n_accumulated_iters = n_init_iter
	while X.shape[0] > 0:
		if bs < X.shape[0]:
			X_batch, X, y_batch, y = sklearn.model_selection.train_test_split(X, y, train_size=bs,
			                                                                  random_state=random_state, shuffle=True)
			n_update_iter = args['update_iters']
		else:
			X_batch, X, y_batch, y = X, np.zeros((0,)), y, np.zeros((0,))
			left_iters = 1  # 250:750 = 1:3
			_iters = int(np.round(left_iters * (1 / 4)))
			n_update_iter = (_iters, left_iters - _iters)
		if i > 0 and i % 5 == 0:
			is_recompute_P = True
		else:
			is_recompute_P = False
		# is_recompute_P = True   # always recompute P from scratch
		_n_accumulated_iters += sum(n_update_iter)
		print(f'{i}-th batch, X_batch: {X_batch.shape}, y_batch: {collections.Counter(y_batch)}, '
		      f'is_recompute_P: {is_recompute_P}, update_iters: {n_update_iter}, '
		      f'accumulated_iters: {_n_accumulated_iters}')

		# 3.1 Incremental TSNE
		st = time.time()
		inc_tsne.update(X_pre, y_pre, X_batch, y_batch, n_update_iter, is_recompute_P)
		ed = time.time()
		# inc_tsne_duration = res['inc_tsne'][-1][0] + (ed - st)
		inc_tsne_duration = (ed - st)
		inc_tsne_n_iter_ = inc_tsne.n_iter_ + 1 if sum(n_update_iter) > 0 else 0  # start for 0
		X_pre = np.concatenate([X_pre, X_batch], axis=0)
		y_pre = np.concatenate([y_pre, y_batch], axis=0)
		scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
		res['inc_tsne'].append((inc_tsne_duration, inc_tsne_n_iter_, scores))
		X_inc_tsne = inc_tsne.embedding_

		# 3.2 Batch TSNE
		st = time.time()
		tsne.fit(X_pre, y_pre)
		ed = time.time()
		tsne_duration = ed - st
		scores = evaluate(X_pre, y_pre, tsne.embedding_)
		tsne.n_iter_ = tsne.n_iter_ + 1 if tsne.n_iter_ > 0 else 0  # start for 0
		res['tsne'].append((tsne_duration, tsne.n_iter_, scores))
		X_tsne = tsne.embedding_

		trust_diff = trustworthiness(X_tsne, X_inc_tsne, n_neighbors=5, metric='euclidean')
		res['trust_diff'].append(trust_diff)
		if is_show:
			f = os.path.join(out_dir, 'batch', f'batch_{i}.png')
			nrows, ncols = 1, 2
			fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
			print(f'figure number: {plt.gcf().number}, {fig.number}')
			# ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
			n_batch = X_batch.shape[0]
			# generate a list of markers and another of colors
			markers = ["*", "v", "^", "<", ">", "1", '2', '3', '4', '8', 's', 'S', 'p', 'P']
			# colors = get_colors()
			colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'tab:red', 'tab:green', 'tab:blue', 'table:yellow']
			batch_colors = [colors[v.item()] for v in y_pre[-n_batch:]]
			batch_markers = [markers[v.item()] for v in y_pre[-n_batch:]]
			ax[0].scatter(X_tsne[:-n_batch, 0], X_tsne[:-n_batch, 1], c=y_pre[:-n_batch])
			# ax[0].scatter(X_tsne[-n_batch:, 0], X_tsne[-n_batch:, 1], c=batch_colors, marker=batch_markers)
			# for _j in range(X_tsne.shape[0]-n_batch):
			# 	idx_label = y_pre[_j].item()
			# 	ax[0].scatter(X_tsne[_j, 0], X_tsne[_j, 1], marker=markers[idx_label])
			for _j in range(n_batch):
				ax[0].scatter(X_tsne[(-n_batch + _j), 0], X_tsne[(-n_batch + _j), 1], c=batch_colors[_j],
				              marker=batch_markers[_j])
			tsne_duration, tsne_iters, scores = res['tsne'][-1]
			trust = scores['trustworthiness']
			acc = scores['acc_1nn']
			ax[0].set_title(f'TSNE on {data_name}{X_pre.shape} takes {fmt(tsne_duration)}s,\n'
			                f'{tsne_iters} iterations\n'
			                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')

			# ax[1].scatter(X_inc_tsne[:, 0], X_inc_tsne[:, 1], c=y_pre)
			ax[1].scatter(X_inc_tsne[:-n_batch, 0], X_inc_tsne[:-n_batch, 1], c=y_pre[:-n_batch])
			# ax[1].scatter(X_inc_tsne[-n_batch:, 0], X_inc_tsne[-n_batch:, 1], c=batch_colors, marker=batch_markers)
			for _j in range(n_batch):
				ax[1].scatter(X_inc_tsne[(-n_batch + _j), 0], X_inc_tsne[(-n_batch + _j), 1], c=batch_colors[_j],
				              marker=batch_markers[_j])
			inc_tsne_duration, inc_tsne_iters, scores = res['inc_tsne'][-1]
			trust = scores['trustworthiness']
			acc = scores['acc_1nn']
			ax[1].set_title(f'INC_TSNE({Y_update_init}) on batch_{i}{X_batch.shape} takes {fmt(inc_tsne_duration)}s,\n'
			                f'{inc_tsne_iters} iterations, trust_diff: {fmt(trust_diff)}\n'
			                f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
			# fig.suptitle(f'INC_TSNE X: batch_{batch}')
			plt.tight_layout()
			check_path(os.path.dirname(f))
			plt.savefig(f, dpi=600, bbox_inches='tight')
			plt.show()
			plt.close()
			batch_figs.append(f)

		# _show_update_data(tsne, inc_tsne, out_dir, idx=i)

		# time components: previous_time + current_(update_dist_time + update_P_time + concat_Y_time + update_Y_time)
		inc_tsne_duration_components = '+'.join(fmt([res['inc_tsne'][-2][0]] + list(inc_tsne.update_res['time'])))
		tsne_duration_components = '+'.join(fmt([0] + list(tsne.fit_res['time'])))
		inc_tsne_kl = ' - '.join([','.join(fmt(list(v))) for v in inc_tsne.update_res['kl_divergence']])
		tsne_kl = ' - '.join([','.join(fmt(list(v))) for v in tsne.fit_res['kl_divergence']])
		print(f'batch_{i},  tsne({tsne_iters}): {fmt(tsne_duration)}s ({tsne_duration_components}, kl: {tsne_kl}) '
		      f'vs. \n\t'
		      f'inc_tsne({inc_tsne_iters}): {fmt(inc_tsne_duration)}s ({inc_tsne_duration_components}, '
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
		# tsne
		if key == 'duration':
			y = [dur for dur, iters, scores in res['tsne']][1:]
			y_label = 'Duration'
		elif key == 'iteration':
			y = [iters for dur, iters, scores in res['tsne']][1:]
			y_label = 'Iteration'
		else:
			y = [scores[key] for dur, iters, scores in res['tsne']][1:]
			y_label = 'Trustworthiness' if key == 'trustworthiness' else 'ACC_1NN'
		plt.plot(x, y, '-ob', label='tsne')
		# inc_tsne
		if key == 'duration':
			y = [dur for dur, iters, scores in res['inc_tsne']][1:]
		elif key == 'iteration':
			y = [iters for dur, iters, scores in res['inc_tsne']][1:]
		else:
			y = [scores[key] for dur, iters, scores in res['inc_tsne']][1:]
		plt.plot(x, y, '-+g', label='inc_tsne')
		if key == 'trustworthiness':
			y = [v for v in res['trust_diff']][1:]
			plt.plot(x, y, '-+c', label='trust_diff')
		plt.xlabel(f'Batch')
		plt.ylabel(y_label)
		plt.title(f'TSNE vs. INC_TSNE ({data_name}): {y_label}')
		plt.legend()
		plt.tight_layout()
		f = os.path.join(out_dir, f'TSNE_vs_INCTSNE-{y_label}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		# plt.show()
		plt.close()

	# Make animation with all batches
	out_file = os.path.join(out_dir, 'all.mp4')
	animate(batch_figs, out_file)
	print(out_file)


if __name__ == '__main__':
	# data_name = '2gaussians'
	# data_name = '2circles'
	# data_name = 's-curve'
	# data_name = '5gaussians-5dims'
	# data_name = '3gaussians-10dims'
	n = 200
	n_iter = 200
	args_lst = []
	# """Case 0: 2circles
	# 	It includes 2 clusters, and each has 500 data points in R^2.
	# """
	args = {'data_name': '2circles', 'n': n, 'n_iter': n_iter}
	# args_lst.append(args)
	"""Case 1: 3gaussians-10dims
		It includes 3 clusters, and each has 500 data points in R^10.
	"""
	args = {'data_name': '3gaussians-10dims', 'n': n, 'n_iter': n_iter}
	# args_lst.append(args)
	"""Case 2: mnist
		It includes 10 clusters, and each has 500 data points in R^784.
	"""
	args = {'data_name': 'mnist', 'n': n, 'n_iter': n_iter}
	args_lst.append(args)
	for args in args_lst:
		for update_init in ['weighted']:  # 'Gaussian',
			perplexity = 50
			args['method'] = "exact"  # 'exact', "barnes_hut"
			args['perplexity'] = perplexity
			args['update_init'] = update_init
			sub_dir = '|'.join([args['method'], f'{perplexity}', args['update_init']])
			args['out_dir'] = os.path.join('out', args['data_name'], str(args['n']), sub_dir)
			print(f'\n\nargs: {args}')
			main(args)
