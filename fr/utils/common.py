import os


def check_path(dir_path):

	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	