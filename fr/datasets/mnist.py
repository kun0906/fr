"""
	1. Download from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download
	2. https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf
		(Introduce MNIST details)

"""
import collections
import copy
import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def mnist_diff_sigma_n(args, random_state=42):
	"""
	Parameters
	----------
	args
	random_state

	Returns
	-------

	"""
	n_clients = args['N_CLIENTS']
	dataset_detail = args['DATASET']['detail']  # 'nbaiot_user_percent_client:ratio_0.1'
	p1 = dataset_detail.split(':')
	ratio = float(p1[1].split('_')[1])

	p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
	p1_0_c1 = p1_0[0].split('-')
	n1 = int(p1_0_c1[0].split('_')[1])

	# tmp = p1_0_c1[1].split('_')
	# sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

	in_dir='datasets/MNIST/mnist'
	IS_PCA = args['IS_PCA']
	# TODO: update IS_CNN in the future, currently use IS_PCA also for CNN
	# IS_CNN = False if 'IS_CNN' not in args.keys() else args['IS_CNN']
	if type(IS_PCA) == str and 'CNN' in str(IS_PCA):
		IS_CNN = True
	else:
		IS_CNN = False

	if IS_PCA == True:
		out_file = 'Xy_PCA.data'
	elif IS_CNN:
		out_file = 'Xy_CNN.data'
	else:
		out_file = 'Xy.dat'
	out_file = os.path.join(in_dir, out_file)

	if random_state == 0:
		# i_repeat * 10 = 0 and avoid some wrong data, so for the first seed, we just delete the previous data.
		if os.path.exists(out_file):
			os.remove(out_file)

	if IS_PCA == True:
		if not os.path.exists(out_file):
			file = os.path.join(in_dir, 'mnist_train.csv')
			df = pd.read_csv(file)
			y = df.label.values
			normalized_method = args['NORMALIZE_METHOD']
			if not normalized_method in ['std']:
				msg = f'is_pca: {IS_PCA}, however, NORMALIZE_METHOD: {normalized_method}'
				raise ValueError(msg)
			else:
				X = copy.deepcopy(df.iloc[:, 1:].values)
				std = sklearn.preprocessing.StandardScaler()
				std.fit(X)
				X = std.transform(X)

			pca = sklearn.decomposition.PCA(n_components=0.95)
			pca.fit(X)
			print(f'pca.explained_variance_ratio_:{pca.explained_variance_ratio_}')
			X_train = pca.transform(X)
			y_train = y

			file = os.path.join(in_dir, 'mnist_test.csv')
			df = pd.read_csv(file)
			if not normalized_method in ['std']:
				msg = f'is_pca: {IS_PCA}, however, NORMALIZE_METHOD: {normalized_method}'
				raise ValueError(msg)
			else:
				X_test = std.transform(df.iloc[:, 1:].values)
			X_test = pca.transform(X_test)
			y_test = df.label.values
			with open(out_file, 'wb') as f:
				pickle.dump((X_train, y_train, X_test, y_test), f)
		else:
			with open(out_file, 'rb') as f:
				X_train, y_train, X_test, y_test = pickle.load(f)
	elif IS_CNN == True:
		if not os.path.exists(out_file):
			# for training set
			file = os.path.join(in_dir, 'mnist_train.csv')
			df = pd.read_csv(file)
			X = df.iloc[:, 1:].values
			y = df.label.values
			normalized_method = args['NORMALIZE_METHOD']

			""" Please run fit() on GPU if you want save yourself.
			Commands: 
				### Create GPU environment
				ssh ky8517@tigergpu.princeton.edu
				$ module load anaconda3/2021.11
				$ conda create --name tf2_10_0-gpu-py397 python=3.9.7
				$ conda env list 
				$ conda activate tf2_10_0-gpu-py397
				$ cd /scratch/gpfs/ky8517/fkm
				$ pip3 install -r requirement.txt       (install on the login machine)
		
				### pip install tensorflow-gpu==2.10.0 --no-cache-dir
				### conda deactivate
				
				### log into a GPU node. 
				$ srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
				$ cd /scratch/gpfs/ky8517/fkm/fkm 
				### module purge
				$module load anaconda3/2021.11
				$source activate tf2_10_0-gpu-py397
				
				# check cuda and cudnn version for tensorflow_gpu==1.13.1
				# https://www.tensorflow.org/install/source#linux
				module load cudatoolkit/11.2
				module load cudnn/cuda-11.x/8.2.0
				
				whereis nvcc
				which nvcc
				nvcc --version
				
				cd /scratch/gpfs/ky8517/fkm/fkm 
				PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 datasets/mnist.py 
				
				### download the fitted model from HPC to local 
				scp ky8517@tigergpu.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/datasets/MNIST/vgg16_zero.h5 ~/Downloads/

			"""
			out_dim = 100
			model_file = os.path.join(in_dir,  f'vgg16_zero_d_{out_dim}.h5')
			OVERWRITE = False
			if OVERWRITE and os.path.exists(model_file): os.remove(model_file)
			from fkm.datasets.vgg16_zero import gen_cnn_features, show_img, resize_img, report
			from keras.utils import to_categorical
			# show_img(X[0].reshape((28, 28)), root_dir = in_dir)  # for debug
			X = X / 255
			# show_img(X[0].reshape((28, 28)), root_dir = in_dir)  # for debug
			if not os.path.exists(model_file):

				import random
				seed = 42
				# optional
				# for numpy.random
				np.random.seed(seed)
				# for built-in random
				random.seed(seed)
				# for hash seed
				os.environ["PYTHONHASHSEED"] = str(seed)
				import tensorflow as tf
				tf.random.set_seed(seed)

				from fkm.datasets.vgg16_zero import VGG16_MODEL, resize_img, plot_training, report
				y_onehot = to_categorical(y)
				model = VGG16_MODEL(n_classes=10, out_dim = out_dim)
				# with tensorflow.device('/gpu:0'): # Keras default uses gpu if it's available
				print('gpu_name: ', tf.test.gpu_device_name(), ', is_gpu: ', tf.test.is_gpu_available())
				X_train = resize_img(X) # for vgg input shape (32, 32, 3)
				# fit a model from scratch because we do not freeze all the layers
				history = model.fit(X_train, y_onehot, epochs=10, batch_size=128, verbose=True, validation_data=(X_train, y_onehot))
				print("Fitting the model completed.")
				print('\nPlotting training results.')
				plot_training(history, root_dir=in_dir)

				print('\nTraining report.')
				report(model, X_train, y_onehot)
				# print('\nTesting report.')
				# report(model, xtest, ytest)
				# dump model to disk
				model.save(model_file)
			# del model  # deletes the existing model
			else:
				import keras
				# Reload the model
				model = keras.models.load_model(model_file)

			# model summary
			model.summary()
			print('\nReport on the training set.')
			report(model, resize_img(X), to_categorical(y))
			X_train = gen_cnn_features(model, X)
			y_train = y
			print(f'X_train: {X_train.shape}')

			# for testing test
			file = os.path.join(in_dir, 'mnist_test.csv')
			df = pd.read_csv(file)
			X = df.iloc[:, 1:].values
			y = df.label.values
			X = X/255
			print('Report on the testing set.')
			report(model, resize_img(X), to_categorical(y))
			X_test = gen_cnn_features(model, X)
			y_test = y
			print(f'X_test: {X_test.shape}')
			with open(out_file, 'wb') as f:
				pickle.dump((X_train, y_train, X_test, y_test), f)
		else:
			with open(out_file, 'rb') as f:
				X_train, y_train, X_test, y_test = pickle.load(f)
	else:
		file = os.path.join(in_dir, 'mnist_train.csv')
		df = pd.read_csv(file)
		X_train =  df.iloc[:, 1:].values
		y_train = df.label.values

		file = os.path.join(in_dir, 'mnist_test.csv')
		df = pd.read_csv(file)
		X_test = df.iloc[:, 1:].values
		y_test = df.label.values

	print(f'raw_X_train: {X_train.shape}')
	def get_xy():
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []
		n_data_points = 0

		# print('original train:', collections.Counter(df.label))
		for y_i in sorted(set(y_train)):
			indices = np.where(y_train==y_i)
			X_train_, X_, y_train_, y_ = train_test_split(X_train[indices], y_train[indices], train_size=n1, shuffle=True,
			                                            random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(X_train_)  # each client has one user's data
			clients_train_y.append(y_train_)

		for y_i in sorted(set(y_test)):
			indices = np.where(y_test == y_i)
			X_test_, _, y_test_, _ = train_test_split(X_test[indices], y_test[indices], train_size=2,
			                                              shuffle=True,
			                                              random_state=random_state)  # train set = 1-ratio

			clients_test_x.append(X_test_)  # each client has one user's data
			clients_test_y.append(y_test_)

		return clients_train_x, clients_train_y, clients_test_x, clients_test_y

	clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy()

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
	print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
	return x, labels


if __name__ == '__main__':
	# mnist_diff_sigma_n({'N_CLIENTS': 0, 'IS_PCA':True, 'DATASET': {'detail': 'n1_200:ratio_0.1'}})
	mnist_diff_sigma_n({'N_CLIENTS': 0, 'IS_PCA': 'CNN', 'IN_DIR': './datasets', 'NORMALIZE_METHOD': 'std',
	                    'DATASET': {'name': 'MNIST', 'detail': 'n1_200:ratio_0.1'}})
