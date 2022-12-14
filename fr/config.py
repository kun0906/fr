""" Parse the config file

	# Use a dictionary to configure the project directly
	# instead of methods (e.g., ini and yaml) that ultimately require you to parse the configuration file into a dictionary.

"""
# Email: kun.bj@outlook.com

import argparse
import os.path
import traceback

import ruamel.yaml as ryaml


def parser(config_file):
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
	                    default=('config.yaml' if not config_file else config_file))
	args_ = parser.parse_args()

	myyaml = ryaml.YAML()
	with open(args_.config_file, "r") as stream:
		try:
			args = myyaml.load(stream)
			args['config_file'] = args_.config_file
		except ryaml.YAMLError as exc:
			traceback.print_exc()
	# pprint(args, sort_dicts=False)

	return args


def load(config_file):
	"""
		laod "args" from config_file

	Parameters
	----------
	config_file

	Returns
	-------

	"""

	myyaml = ryaml.YAML()
	with open(config_file, "r") as f:
		try:
			args = myyaml.load(f)
		except ryaml.YAMLError as exc:
			print(exc)

	return args


def dump(config_file, args):
	"""
		Write "args" to a new config_file

	Parameters
	----------
	config_file
	args

	Returns
	-------

	"""
	config_file = os.path.abspath(config_file)
	out_dir = os.path.dirname(config_file)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	myyaml = ryaml.YAML()
	with open(config_file, "w") as f:
		try:
			myyaml.dump(args, f)
		except ryaml.YAMLError as exc:
			print(exc)

	return config_file
