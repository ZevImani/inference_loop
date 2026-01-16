#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-01:30      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=16000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zmcmc_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zmcmc_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 	# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 


conda run -n ldm python3 sample_mc.py  