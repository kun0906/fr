#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tiger.princeton.edu
#ssh ky8517@tigergpu.princeton.edu
cd /scratch/gpfs/ky8517/fr/fr
module load anaconda3/2021.11
#conda create --name py3104 python=3.10.4
conda activate py3104
#pip3 install -r requirement.txt

srun --nodes=1 --ntasks-per-node=1 --time=2:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#cd /scratch/gpfs/ky8517/fr/fr
#module load anaconda3/2021.11

# python3 main_inc_tsne2.py
#sshfs ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fr/fr ~/tiger
#!/bin/bash

# sshfs ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm tiger
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fr/fr/out ~/Downloads/
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/out/xlsx ~/Downloads/
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results/3GAUSSIANS-Client_epochs_1-n1_0-ratio_0.xlsx ~/Downloads/
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results/3GAUSSIANS-Client_epochs_1-n1_100.xlsx ~/Downloads/
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results/5GAUSSIANS-Client_epochs_1.xlsx ~/Downloads/
#rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results/2GAUSSIANS-Client_epochs_1.xlsx ~/Downloads/

