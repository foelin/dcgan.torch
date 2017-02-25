#! /bin/zsh
export CUDA_VISIBLE_DEVICES=$1
DATA_ROOT=dataset/celebA dataset=folder niter=10000 batchSize=32 nThreads=2 display=1 port=8000 lr=5e-5 th main_wgan.lua