#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=25G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=16G

# Set the name of the job.
#$ -N B150_0.7_BERT_LSTM

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucabdc3/Scratch/output

# Change into temporary directory to run work
cd /lustre/scratch/scratch/ucabdc3/bert_lstm_attack

# load the cuda module (in case you are running a CUDA program)
module unload compilers mpi  
module load compilers/gnu/4.9.2  
module load python3/recommended
module load cuda/10.0.130/gnu-4.9.2  
module load cudnn/7.4.2.24/cuda-10.0
module load tensorflow/2.0.0/gpu-py37

# Run the application - the line below is just a random example.
python train_150.py --freeze=true --nlayer=1 --data=imdb --kept_prob_dropout=0.7 --epoches=50 --learning_rate=0.001 --bidirection=False --tokenizer=bert --save_path=/lustre/scratch/scratch/ucabdc3/bert_lstm_attack/model_params


# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
tar zcvf /lustre/home/ucabdc3/Scratch/files_from_job_bert_train.tar.gz 

# Make sure you have given enough time for the copy to complete!