""" Run this main file for all experiments

	#ssh ky8517@tigergpu.princeton.edu
	srun --time=2:00:00 --pty bash -i
	srun --nodes=1  --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
	#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
	cd /scratch/gpfs/ky8517/rkm/rkm
	module load anaconda3/2021.11

	# check why the process is killed
	dmesg -T| grep -E -i -B100 'killed process'

"""
# Email: kun.bj@outllok.com
import copy
import os
import shutil
import traceback
from pprint import pprint

import numpy as np

from config import parser, dump
from utils.common import check_path

np.set_printoptions(precision=3, suppress=True)

def gen_all_sh(args):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""
	OUT_DIR = f'out/sh'
	data_name = args['data_name']
	n = args['n']
	n_iter = args['n_iter']
	update_iters = args['update_iters']
	job_name = f'{data_name}-{n}-{update_iters}-{n_iter}'
	job_name = job_name.replace(' ', '')
	out_config_file = f'{OUT_DIR}/{job_name}/config.yaml'
	print(out_config_file)
	out_dir = os.path.dirname(out_config_file)
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	dump(out_config_file, args)
	# job_name = f'{dataset_name}-{dataset_detail}'
	# # tmp_dir = '~tmp'
	# # if not os.path.exists(tmp_dir):
	# # 	os.system(f'mkdir {tmp_dir}')
	# if '2GAUSSIANS' in dataset_name:
	# 	t = 48
	# # elif 'FEMNIST' in dataset_name and 'greedy' in algorithm_py_name:
	# # 	t = 48
	# else:
	t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt
### SBATCH --mail-type=end          # send email when job ends
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification. 
## SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
## SBATCH --nodes=1                # node count
## SBATCH --ntasks=1               # total number of tasks across all nodes
## SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
## SBATCH --mem=40G\
### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-user=kun.bj@cloud.com # not work 
### SBATCH --mail-user=<YourNetID>@princeton.edu\

module purge
cd /scratch/gpfs/ky8517/fr/fr
module load anaconda3/2021.11
#conda create --name py3104 python=3.10.4
conda activate py3104
#pip3 install -r requirement.txt

pwd
python3 -V
"""
	# content += "\nexport PYTHONPATH=\"${PYTHONPATH}:..\" \n"
	out_dir_ = f'{out_dir}'
	content += f"\nPYTHONPATH='.' PYTHONUNBUFFERED=TRUE python3 main_inc_tsne2.py --config_file '{out_config_file}' > '{out_dir_}/log.txt' 2>&1  &\n"

	content += "\nwait\n"  # must has this line
	# The bash wait command is a Shell command that waits for background running processes to complete and returns the exit status.
	# Without any parameters, the wait command waits for all background processes to finish before continuing the script.
	content += "echo $!\n"  # stores the background process PID
	content += "echo $?\n"  # $? stores the exit status.
	content += "\necho \'done\n"
	# sh_file = f'{OUT_DIR}/{dataset_name}-{dataset_detail}-{algorithm_name}-{algorithm_detail}.sh'
	sh_file = f'{out_dir}/sbatch.sh'
	check_path(os.path.dirname(sh_file))
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f"sbatch '{sh_file}'"
	print(cmd)
	os.system(cmd)


if __name__ == '__main__':
	i = 1
	for data_name in ['2gaussians', '3gaussians-10dims', 'mnist']:
		for n in [50, 300, 600]:
			for update_iters in [(0, 0), (0, 1), (3, 7), (12, 38)]:
				for n_iter in [100, 1000]:
					print(f'\n*i:{i}')
					args = parser(config_file = 'config.yaml')
					args['data_name'] = data_name
					args['n'] = n
					args['n_iter'] = n_iter
					args['update_iters'] = str(update_iters)
					gen_all_sh(args)
					i+=1
	print(f'\n***Total cnt: {i-1}')
