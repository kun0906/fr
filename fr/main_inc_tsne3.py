"""Incremental TSNE


	Compile perplexity
    cd fr/perplexity
    pip install .   # method 1 (recommended)

	PYTHONPATH=. python3 main_inc_tsne2.py
"""
# Author: kun.bj@outlook.com

import collections
import copy
import os
import pickle
import shutil
import time
import traceback
import warnings
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
from sklearn.metrics import pairwise_distances

import config
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


def show_init_params(out_dir='out', data_name='2gaussians', perplexity=30, data_size=100,
                     update_str='(25, 75)|(25, 75)',
                     init_percents=[0.1, 0.3, 0.5, 0.7, 0.9, 0.999], random_states = [42],
                     method='exact', update_init='weighted', is_show=True):
	cols = 7
	fig, axes = plt.subplots(1, cols, figsize=(20, 4))  # (width, height)
	axes = axes.reshape((1, cols))
	# mu_score = 0
	for i, metric_name in enumerate(
			['perplexity', 'trustworthiness', 'spearman', 'pearson', 'neighborhood_hit', 'normalized_stress', 'kl']):
		tsne_res = []
		inc_tsne_res = []
		for init_percent in init_percents:
			_tsne_res = []
			_inc_tsne_res = []
			for random_state in random_states:
				try:
					out_dir_new = f'{out_dir}/{data_name}/{data_size}/{method}|{perplexity}|{update_init}|seed_{random_state}/{init_percent}|{update_str}'
					res_file = os.path.join(out_dir_new, 'res.out')
					with open(res_file, 'rb') as f:
						res = pickle.load(f)
					n_iter = res['tsne'][-1][1]
					X_shape = res['tsne'][-1][3]['X_shape']
				except Exception as e:
					print(e)
					continue
				if metric_name == 'perplexity':
					try:
						v1 = res['tsne'][-1][3][metric_name]
						v2 = res['inc_tsne'][-1][3][metric_name]
					except Exception as e:
						print('bs(batch_size) < 1, so no online results.')
						# skip this value.
						continue
				elif metric_name == 'kl':
					v1 = res['tsne'][-1][3][metric_name]
					v2 = res['inc_tsne'][-1][3][metric_name]
				elif metric_name == 'spearman':
					v1 = res['tsne'][-1][2][metric_name].correlation
					v2 = res['inc_tsne'][-1][2][metric_name].correlation
				elif metric_name == 'pearson':
					v1 = res['tsne'][-1][2][metric_name].statistic
					v2 = res['inc_tsne'][-1][2][metric_name].statistic
				else:
					v1 = res['tsne'][-1][2][metric_name]
					v2 = res['inc_tsne'][-1][2][metric_name]
				_tsne_res.append(v1)
				_inc_tsne_res.append(v2)
			if len(_tsne_res) > 0:
				tsne_res.append((np.mean(_tsne_res), np.std(_tsne_res)))
				inc_tsne_res.append((np.mean(_inc_tsne_res), np.std(_inc_tsne_res)))
			else:
				continue
		# the last two points (e.g., 0.995, 0.999) will overlap.
		print(init_percent, tsne_res, inc_tsne_res)
		# plt.plot(init_percents, tsne_res, '-+b', label='tsne')
		# plt.plot(init_percents, inc_tsne_res, '-og', label='inc_tsne')
		# plt.xlabel(f'Initial percentage of data')
		# plt.ylabel(f'{metric_name}')
		# plt.title(f'{data_name}: {metric_name}')
		# plt.legend()
		# plt.tight_layout()
		# f = os.path.join(out_dir, f'TSNE_vs_INCTSNE-inits-{metric_name}.png')
		# plt.savefig(f, dpi=600, bbox_inches='tight')
		# if is_show: plt.show()
		# plt.close()
		m = len(tsne_res)
		# axes[0, i].plot(init_percents[:m], tsne_res, '-+b', label='tsne')
		# axes[0, i].plot(init_percents[:m], inc_tsne_res, '-og', label='inc_tsne')
		x = init_percents[:m]
		y, yerr = zip(*tsne_res[:m])
		axes[0, i].errorbar(x, y, yerr=yerr, color='b', marker='*', label='tsne', alpha=0.8, ecolor='r', lw=2, capsize=5, capthick=2)
		y, yerr = zip(*inc_tsne_res[:m])
		axes[0, i].errorbar(x, y, yerr=yerr, color='g', marker='o',label='inc_tsne', alpha=0.8, ecolor='m', lw=2, capsize=5, capthick=2)
		axes[0, i].set_xlabel(f'Initial percentage of data')
		axes[0, i].set_ylabel(f'{metric_name}')
		axes[0, i].set_title(f'{metric_name}')
		axes[0, i].legend()
	fig.suptitle(f'{data_name}, X{X_shape}, n_iter({n_iter}), n_repeats({len(random_states)})')
	plt.tight_layout()
	f = os.path.join(out_dir_new, f'TSNE_vs_INCTSNE-inits.png')
	print(f)
	check_path(os.path.dirname(f))
	plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show: plt.show()
	plt.close()


@timer
def show_avg_local_error(out_dir='out', data_name='2gaussians', perplexity=30, data_size=100,
                         update_str='(25, 75)|(25, 75)',
                         init_percents=[0.1, 0.3, 0.5, 0.7, 0.9],  random_states = [42],
                         method='exact', update_init='weighted', is_show=True):
	fig, axes = plt.subplots(len(init_percents), 8, figsize=(27, 20))  # (width, height)
	for i, init_percent in enumerate(init_percents):
		random_state = random_states[0]
		out_dir_new = f'{out_dir}/{data_name}/{data_size}/{method}|{perplexity}|{update_init}|seed_{random_state}/{init_percent}|{update_str}'
		res_file = os.path.join(out_dir_new, 'res.out')
		try:
			with open(res_file, 'rb') as f:
				res = pickle.load(f)
			n_iter = res['tsne'][-1][1]
			X_shape = res['tsne'][-1][3]['X_shape']
		except Exception as e:
			print(e)
			continue
		print(i, res_file)
		tsne_embedding, inc_tsne_embedding = res['embedding']

		col = 0
		for j, (method_name, embedding) in enumerate(
				[('tsne', tsne_embedding), ('inc_tsne', inc_tsne_embedding), ('inc_tsne_init', inc_tsne_embedding),
				 ('inc_tsne_new', inc_tsne_embedding)]):
			# https://stackoverflow.com/questions/53838301/add-a-verticle-line-between-matplotlib-subplots
			if j == 2:  # add vertical line
				line = plt.Line2D((.5, .5), (.04, .95), color="k", linewidth=3)
				fig.add_artist(line)

			X, y = res['data']
			n, _ = X.shape
			if method_name == 'inc_tsne_init':  # only show the initial embeddings
				n_init = int(np.round(n * init_percent))
				X, embedding = X[:n_init, :], embedding[:n_init, :]
			elif method_name == 'inc_tsne_new':  # only show the new embeddings
				n_init = int(np.round(n * init_percent))
				X, embedding = X[n_init:, :], embedding[n_init:, :]
			else:
				pass
			n, _ = X.shape
			if n < 1: continue
			dist_X = pairwise_distances(X, X)
			# dist_Y = pairwise_distances(tsne_embedding, tsne_embedding)
			dist_Y = pairwise_distances(embedding, embedding)

			# plt.scatter(dist_X, dist_Y, c=dist_Y)
			# plt.xlabel(f'distance X')
			# plt.ylabel(f'distance Y')
			# plt.title(f'Initial percentage:{init_percent}, {method_name}')
			# # plt.legend()
			# plt.tight_layout()
			# f = os.path.join(out_dir, f'distance-{init_percent}-{method_name}.png')
			# plt.savefig(f, dpi=600, bbox_inches='tight')
			# if is_show: plt.show()
			# plt.close()

			axes[i, col + 0].scatter(dist_X, dist_Y)
			axes[i, col + 0].set_xlabel('distance in $R^d$')
			axes[i, col + 0].set_ylabel('distance in $R^2$')
			if method_name == 'inc_tsne_init':
				pearson = res['inc_tsne'][0][2]['pearson'].statistic
			# spearman = res['inc_tsne'][0][-1]['spearman'].correlation
			elif method_name == 'inc_tsne_new':
				pearson = res['inc_tsne'][1][2]['pearson'].statistic
			# spearman = res['inc_tsne'][1][-1]['spearman'].correlation
			else:
				pearson = res[method_name][-1][2]['pearson'].statistic
			# spearman = res[method_name][-1][-1]['spearman'].correlation
			if method_name == 'tsne':
				axes[i, col + 0].set_title(f'{method_name}: pearson:{fmt(pearson)}')
			elif method_name == 'inc_tsne_init':
				axes[i, col + 0].set_title(f'initial X{X.shape}, pearson:{fmt(pearson)}')
			elif method_name == 'inc_tsne_new':
				axes[i, col + 0].set_title(f'New X{X.shape}')
			else:
				axes[i, col + 0].set_title(f'{method_name}({init_percent}): pearson:{fmt(pearson)}')
			# plt.xlabel(f'distance X')
			# plt.ylabel(f'distance Y')
			# plt.title(f'Initial percentage:{init_percent}, {method_name}')

			M_x = []
			for _i in range(n):
				mx = 0
				max_x = max(dist_X[_i])
				max_y = max(dist_Y[_i])
				for _j in range(n):
					if _i == _j: continue
					mx += abs(dist_X[_i][_j] / max_x - dist_Y[_i][_j] / max_y)
				if n - 1 > 0:
					M_x.append(1 / (n - 1) * mx)
				else:
					M_x.append(mx)

			# y_label = 'average local error'
			# plt.scatter(embedding[:, 0], embedding[:, 1], c=M_x)
			# # plt.plot(init_percents, inc_tsne_res, '-og', label='inc_tsne')
			# plt.xlabel(f'Initial percentage of data')
			# plt.ylabel(f'{y_label}')
			# plt.title(f'{data_name}: Initial percentage: {init_percent}, {embedding.shape}, {method_name}')
			# # plt.legend()
			# plt.tight_layout()
			# f = os.path.join(out_dir, f'{y_label}-{init_percent}-{method_name}.png')
			# plt.savefig(f, dpi=600, bbox_inches='tight')
			# if is_show: plt.show()
			# plt.close()

			if method_name == 'inc_tsne_init':
				train_time = res['inc_tsne'][0][0]
			elif method_name == 'inc_tsne_new':
				train_time = res['inc_tsne'][1][0] - res['inc_tsne'][0][0]  # not includes the initial time
			else:
				train_time = res[method_name][-1][0]

			axes[i, col + 1].set_title(f'{fmt(train_time)}s, X{X.shape}')
			# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html
			pcm = axes[i, col + 1].scatter(embedding[:, 0], embedding[:, 1], c=M_x)
			fig.colorbar(pcm, ax=axes[i, col + 1])
			# if data_name == '3gaussians-10dims':
			# 	axes[i, col + 1].set_xlim([-25, 25])
			# 	axes[i, col + 1].set_ylim([-20, 20])
			col += 2

	fig.suptitle(f'{data_name}, X{X_shape}, n_iter({n_iter})')
	# rect=(left, bottom, right, top), default: (0, 0, 1, 1)
	plt.tight_layout(rect=(0, 0, 1, 0.98))
	y_label = 'average_local_error'
	f = os.path.join(out_dir_new, f'{y_label}.png')
	print(f)
	check_path(os.path.dirname(f))
	plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show: plt.show()
	plt.close()


@timer
def main(args_raw):
	is_show = False
	data_name = args_raw['data_name']
	"""1. Generate dataset 
		
	"""
	X_raw, y_raw = gen_data.gen_data(n=args_raw['n'], data_type=args_raw['data_name'], is_show=True,
	                                 with_noise=False, random_state=42)
	X_raw, y_raw = sklearn.utils.shuffle(X_raw, y_raw, random_state=42)
	print(f'X.shape: {X_raw.shape}, y: {collections.Counter(y_raw)}')
	print(np.quantile(X_raw, q=[0, 0.25, 0.5, 0.75, 1.0], axis=0))

	if args_raw['data_name'] == 'mnist':
		indices = np.where((y_raw % 2 == 0) & (y_raw > 0))[0]
		X_raw = X_raw[indices]
		y_raw = y_raw[indices]

	n, d = X_raw.shape
	for init_idx, init_percent in enumerate(args_raw['init_percents']):
		random_state = args_raw['random_state']
		print(f'\n\n*init_percent:{init_percent}')
		res = {'tsne': [], 'inc_tsne': []}
		X, y = copy.deepcopy(X_raw), copy.deepcopy(y_raw)
		indices = np.arange(n)
		args = copy.deepcopy(args_raw)
		n_iter = args_raw['n_iter']
		n_init = int(np.round(n * init_percent))  # the total number of initial training data size.
		args['perplexity'] = min(args['perplexity'], n_init - 1)
		n_init_iter = n_iter  # int(np.round(n_iter * 1.0))  # the total number of initial iterations
		n1 = int(np.round(n_init_iter * (1 / 4)))  # 250:750 = 1:3
		args['init_iters'] = (n1, n_init_iter - n1)

		# batch_percent = 1.0
		# bs = int(np.round((n - n_init) * batch_percent))  # bs is 10% of (n-n_init)
		# each_iter = int(0.5 * n_iter)  # use few iterations for online update
		# n1 = int(np.round(each_iter * (1 / 4)))  # 250:750 = 1:3
		# # print(f'n1: {n1}')
		# if n1 <= 0 or each_iter - n1 <= 0:
		# 	raise ValueError(f'n_iteration {n_iter} is too small!')
		# args['update_iters'] = (n1, n_iter - n1)  # when i%5 != 0, learning stage 1 and 2
		# args['update_iters'] = (0, 1)
		bs = n - n_init
		if bs < 1:
			warnings.warn(f'bs: {bs}')
			continue
		sub_dir = '|'.join([str(init_percent), str(args['init_iters']), str(args['update_iters'])])
		out_dir = os.path.join(args['out_dir'], sub_dir)
		if os.path.exists(out_dir):
			shutil.rmtree(out_dir)
		print(f'n: {n}, n_init: {n_init}, bs: {bs}, args: {args}')

		"""2. Get the initial fitting results for TSNE and INC_TSNE
		"""
		# Incremental TSNE
		Y_update_init = args['update_init']
		tsne = TSNE(perplexity=args['perplexity'], method=args['method'], n_iter=n_iter,
		            _EXPLORATION_N_ITER=int(np.round(n_iter * (1 / 4))),
		            random_state=random_state, verbose=1)
		inc_tsne = INC_TSNE(perplexity=args['perplexity'], method=args['method'], update_init=args['update_init'],
		                    n_iter=n_iter, init_iters=args['init_iters'], update_iters=args['update_iters'],
		                    random_state=random_state, is_last_batch=False, verbose=1)

		# if init_idx == 0:
		# 	X_pre, X, y_pre, y, indices_pre, indices = sklearn.model_selection.train_test_split(X, y, indices, train_size=n_init, stratify=y,
		#                                                               random_state=random_state, shuffle=True)
		# 	X0, y0, n_init0 = copy.deepcopy(X_pre), copy.deepcopy(y_pre), n_init
		# 	indices0 = copy.deepcopy(indices_pre)
		# else:
		# 	indices = np.setdiff1d(indices, indices0)
		# 	X, y = X[indices], y[indices]
		# 	# once you use train_test_split, the order of X, y and indices will change and then the tsne result will change as well.
		# 	X_pre, X, y_pre, y, indices_pre, indices = sklearn.model_selection.train_test_split(X, y, indices,
		# 	                                                                                    train_size=n_init-n_init0, stratify=y,
		# 	                                                              random_state=random_state, shuffle=True)
		# 	X_pre = np.concatenate([X0, X_pre], axis=0)
		# 	y_pre = np.concatenate([y0, y_pre], axis=0)
		# 	X0, y0, n_init0 = copy.deepcopy(X_pre), copy.deepcopy(y_pre), n_init
		# 	indices0 = np.concatenate([indices0, indices_pre], axis=0)
		# 	print(f'X0:{X0.shape}, y0:{y0.shape}, indices0:{indices0.shape}')
		# continue
		X_pre, X, y_pre, y = X[:n_init, :], X[n_init:, :], y[:n_init], y[n_init:]

		# Incremental TSNE for initializing training phase.
		st = time.time()
		inc_tsne.fit(X_pre, y_pre)
		ed = time.time()
		inc_tsne_duration = ed - st
		inc_tsne.n_iter_ += 1  # start for 0
		print(f'inc_tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(inc_tsne_duration)}s')
		scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
		inc_tsne_params = {'perplexity': args['perplexity'], 'kl': inc_tsne.kl_divergence_, 'X_shape': X_pre.shape}
		res['inc_tsne'].append((inc_tsne_duration, inc_tsne.n_iter_, scores, inc_tsne_params))
		X_inc_tsne = inc_tsne.embedding_
		# TSNE
		st = time.time()
		tsne.fit(X_pre, y_pre)
		ed = time.time()
		tsne_duration = ed - st
		tsne.n_iter_ += 1  # start for 0
		print(f'tsne initial fitting with {inc_tsne.n_iter_} iterations: {fmt(tsne_duration)}s')
		scores = evaluate(X_pre, y_pre, tsne.embedding_)
		tsne_params = {'perplexity':args['perplexity'], 'kl':tsne.kl_divergence_, 'X_shape': X_pre.shape}
		res['tsne'].append((tsne_duration, tsne.n_iter_, scores, tsne_params))
		X_tsne = tsne.embedding_

		trust_diff = trustworthiness(X_tsne, X_inc_tsne, n_neighbors=5, metric='euclidean')
		res['trust_diff'] = [trust_diff]
		# if is_show:
		# 	show_embedded_data(tsne, inc_tsne, res, X_pre, y_pre, args, out_dir, idx=0)
		#
		# 	# save update data
		# 	_show_update_data(tsne, inc_tsne, out_dir, idx=0)
		#
		# 	show_detail_flg = False
		# 	if show_detail_flg:
		# 		# show each update iteration
		# 		idx_batch = 0
		# 		figs = []
		# 		for learning_phase in [0, 1]:  # two phases of gradient updates with/without early_exaggeration
		# 			# _n = min(5, len(tsne._update_data[learning_phase]), len(inc_tsne._update_data[learning_phase]))
		# 			_n = max(10, len(tsne._update_data[learning_phase]), len(inc_tsne._update_data[learning_phase]))
		# 			for j in range(_n):
		# 				fig = _show_embedded_data(tsne, inc_tsne, X_pre, y_pre, args, out_dir, idx=idx_batch,
		# 				                          learning_phase=learning_phase, jth_iter=j)
		# 				figs.append(fig)
		# 		animate(figs, out_file=os.path.join(out_dir, 'batch', f'{idx_batch}.mp4'))

		"""3. Get the results for each batch
		"""
		res_file = os.path.join(out_dir, 'res.out')
		i = 1
		batch_figs = []
		_n_accumulated_iters = n_init_iter
		while X.shape[0] > 0:  # in this case (for evaluating the effect of subsampling size), we only have 1 batch.
			if bs < X.shape[0]:
				# X_batch, X, y_batch, y = sklearn.model_selection.train_test_split(X, y, train_size=bs,
				#                                                                   random_state=random_state,
				#                                                                   shuffle=True)
				n_update_iter = args['update_iters']
				break
			else:
				X_batch, X, y_batch, y = X, np.zeros((0,)), y, np.zeros((0,))
				# left_iters = 10  # 250:750 = 1:3
				# _iters = int(np.round(left_iters * (1 / 4)))
				# n_update_iter = (_iters, left_iters - _iters)
				n_update_iter = args['update_iters']
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
			inc_tsne_duration = res['inc_tsne'][-1][0] + (ed - st)
			# inc_tsne_duration = (ed - st)
			inc_tsne_n_iter_ = inc_tsne.n_iter_ + 1 if sum(n_update_iter) > 0 else 0  # start for 0
			X_pre = np.concatenate([X_pre, X_batch], axis=0)
			y_pre = np.concatenate([y_pre, y_batch], axis=0)
			scores = evaluate(X_pre, y_pre, inc_tsne.embedding_)
			inc_tsne_params = {'perplexity': args['perplexity'], 'kl': inc_tsne.kl_divergence_, 'X_shape': X_pre.shape}
			res['inc_tsne'].append((inc_tsne_duration, inc_tsne_n_iter_, scores, inc_tsne_params))
			X_inc_tsne = inc_tsne.embedding_

			# 3.2 Batch TSNE
			print(np.all(X_pre == X_raw), np.all(y_pre == y_raw))
			print(args['perplexity'], args['method'], n_iter, int(np.round(n_iter * (1 / 4))), random_state)
			tsne = TSNE(perplexity=args['perplexity'], method=args['method'], n_iter=n_iter,
			            _EXPLORATION_N_ITER=int(np.round(n_iter * (1 / 4))),
			            random_state=random_state, verbose=0)
			st = time.time()
			tsne.fit(X_pre, y_pre)
			ed = time.time()
			tsne_duration = ed - st
			scores = evaluate(X_pre, y_pre, tsne.embedding_)
			tsne.n_iter_ = tsne.n_iter_ + 1 if tsne.n_iter_ > 0 else 0  # start for 0
			tsne_params = {'perplexity': args['perplexity'], 'kl': tsne.kl_divergence_, 'X_shape': X_pre.shape}
			res['tsne'].append((tsne_duration, tsne.n_iter_, scores, tsne_params))
			X_tsne = tsne.embedding_

			trust_diff = trustworthiness(X_tsne, X_inc_tsne, n_neighbors=5, metric='euclidean')
			res['trust_diff'].append(trust_diff)
			# print(res.items())
			if True:
				f = os.path.join(out_dir, 'batch', f'batch_{i}.png')
				nrows, ncols = 1, 2
				fig, ax = plt.subplots(nrows, ncols, figsize=(10, 5), )  # width, height
				print(f'figure number: {plt.gcf().number}, {fig.number}')
				# ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pre)
				n_batch = X_batch.shape[0]
				# generate a list of markers and another of colors
				markers = ["*", "v", "^", "<", ">", "1", '2', '3', '4', '8', 's', 'S', 'p', 'P']
				# colors = get_colors()
				colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'tab:red', 'tab:green', 'tab:blue',
				          'table:yellow']
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
				tsne_duration, tsne_iters, scores, _ = res['tsne'][-1]
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
				inc_tsne_duration, inc_tsne_iters, scores, _ = res['inc_tsne'][-1]
				trust = scores['trustworthiness']
				acc = scores['acc_1nn']
				ax[1].set_title(
					f'INC_TSNE({Y_update_init}) on batch_{i}{X_batch.shape} takes {fmt(inc_tsne_duration)}s,\n'
					f'{inc_tsne_iters} iterations, trust_diff: {fmt(trust_diff)}\n'
					f'trustworthiness:{fmt(trust)}, acc_1nn: {fmt(acc)}')
				# fig.suptitle(f'INC_TSNE X: batch_{batch}')
				plt.tight_layout()
				check_path(os.path.dirname(f))
				plt.savefig(f, dpi=600, bbox_inches='tight')
				if is_show: plt.show()
				plt.close()
				batch_figs.append(f)

			# _show_update_data(tsne, inc_tsne, out_dir, idx=i)

			# time components: previous_time + current_(update_dist_time + update_P_time + concat_Y_time + update_Y_time)
			inc_tsne_duration_components = '+'.join(fmt([res['inc_tsne'][-2][0]] + list(inc_tsne.update_res['time'])))
			tsne_duration_components = '+'.join(fmt([0] + list(tsne.fit_res['time'])))
			# KL divergence error
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
		res['data'] = (X_pre, y_pre)
		res['embedding'] = (tsne.embedding_, inc_tsne.embedding_)
		print(res_file)
		check_path(os.path.dirname(res_file))
		with open(res_file, 'wb') as f:
			pickle.dump(res, f)


# # Plot final results
# for key in ['duration', 'iteration', 'trustworthiness', 'acc_1nn']:
# 	x = range(len(res['tsne'][1:]))
# 	# tsne
# 	if key == 'duration':
# 		y = [dur for dur, iters, scores in res['tsne']][1:]
# 		y_label = 'Duration'
# 	elif key == 'iteration':
# 		y = [iters for dur, iters, scores in res['tsne']][1:]
# 		y_label = 'Iteration'
# 	else:
# 		y = [scores[key] for dur, iters, scores in res['tsne']][1:]
# 		y_label = 'Trustworthiness' if key == 'trustworthiness' else 'ACC_1NN'
# 	plt.plot(x, y, '-ob', label='tsne')
# 	# inc_tsne
# 	if key == 'duration':
# 		y = [dur for dur, iters, scores in res['inc_tsne']][1:]
# 	elif key == 'iteration':
# 		y = [iters for dur, iters, scores in res['inc_tsne']][1:]
# 	else:
# 		y = [scores[key] for dur, iters, scores in res['inc_tsne']][1:]
# 	plt.plot(x, y, '-+g', label='inc_tsne')
# 	if key == 'trustworthiness':
# 		y = [v for v in res['trust_diff']][1:]
# 		plt.plot(x, y, '-+c', label='trust_diff')
# 	plt.xlabel(f'Batch')
# 	plt.ylabel(y_label)
# 	plt.title(f'TSNE vs. INC_TSNE ({data_name}): {y_label}')
# 	plt.legend()
# 	plt.tight_layout()
# 	f = os.path.join(out_dir, f'TSNE_vs_INCTSNE-{y_label}.png')
# 	plt.savefig(f, dpi=600, bbox_inches='tight')
# 	if is_show: plt.show()
# 	plt.close()
#
# # Make animation with all batches
# out_file = os.path.join(out_dir, 'all.mp4')
# animate(batch_figs, out_file)
# print(out_file)



if __name__ == '__main__':

	st = time.time()
	for n in [50, 300, 600]:
		for update_iters in [(0, 0), (0, 1)]:  # (0, 1), (3, 7), (12, 38)
			for n_iter in [100, 1000]:
				# # data_name = '1gaussian'
				# data_name = '2gaussians'
				# # data_name = '2circles'
				# # data_name = 's-curve'
				# # data_name = '5gaussians-5dims'
				data_name = '3gaussians-10dims'
				perplexity = 30
				# data_size = 300 # 3000,
				# n = 600  # 2 clusters * n = 600 : 0.1* 600 = 60
				# n_iter = 1000  # > 8 hours
				n1 = int(np.round(n_iter * (1 / 4)))  # 250:750 = 1:3
				# update_iters = (0,1)
				update_str = f'({n1}, {n_iter - n1})|{update_iters}'
				init_percents = [0.1, 0.3, 0.5, 0.7, 0.9, 0.995, 0.999]
				random_states = [100, 200, 300, 400, 500]   # [100, 200] #
				# show_init_params('out', data_name, perplexity, n, update_str=update_str,
				#                  init_percents=init_percents, random_states = random_states,
				#                  method='exact', update_init= 'weighted', is_show=True)
				# show_avg_local_error('out', data_name, perplexity, n, update_str= update_str,
				#                  init_percents= init_percents, random_states = random_states,
				#                  method='exact', update_init= 'weighted', is_show=True)
				# exit(0)

				args_lst = []
				# """Case 0: 1circles
				# 	It includes 1 clusters, and each has 500 data points in R^2.
				# """
				args = {'data_name': '1gaussian', 'n': n, 'n_iter': n_iter}
				# args_lst.append(args)

				# """Case 0: 2gaussians
				# 	It includes 2 clusters, and each has 500 data points in R^2.
				# """
				args = {'data_name': '2gaussians', 'n': n, 'n_iter': n_iter}
				# args_lst.append(args)

				# """Case 0: 2circles
				# 	It includes 2 clusters, and each has 500 data points in R^2.
				# """
				args = {'data_name': '2circles', 'n': n, 'n_iter': n_iter}
				# args_lst.append(args)
				"""Case 1: 3gaussians-10dims
					It includes 3 clusters, and each has 500 data points in R^10.
				"""
				args = {'data_name': '3gaussians-10dims', 'n': n, 'n_iter': n_iter}
				args_lst.append(args)
				"""Case 2: mnist
					It includes 10 clusters, and each has 500 data points in R^784.
				"""
				args = {'data_name': 'mnist', 'n': n, 'n_iter': n_iter}
				args_lst.append(args)
				for args in args_lst:
					for update_init in ['weighted']:  # 'Gaussian', 'weighted'
						for random_state in random_states:
							args['random_state'] = random_state
							args['method'] = "exact"  # 'exact', "barnes_hut"
							args['perplexity'] = perplexity
							args['update_init'] = update_init
							args['update_iters'] = update_iters
							args['init_percents'] = init_percents
							sub_dir = '|'.join([args['method'], f'{perplexity}', args['update_init'], f'seed_{random_state}'])
							args['out_dir'] = os.path.join('out', args['data_name'], str(args['n']), sub_dir)
							print(f'\n\nargs: {args}')
							main(args)
						try:
							show_init_params(data_name=args['data_name'], perplexity=perplexity, data_size=n,
							                 update_str=update_str, init_percents=args['init_percents'],
							                 random_states=random_states,
							                 method=args['method'], update_init=args['update_init'], is_show=True)
							show_avg_local_error(data_name=args['data_name'], perplexity=perplexity, data_size=n,
							                     update_str=update_str, init_percents=args['init_percents'],
							                     random_states=random_states,
							                     method=args['method'], update_init=args['update_init'], is_show=True)
						except Exception as e:
							print(e)
							traceback.print_exc()
					# exit(0)
	ed = time.time()
	print(f'\n\nTotal time: {ed - st}s.')
