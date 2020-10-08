#!/bin/bash

#SBATCH --partition=bgmp        ### Partition (like a queue in PBS)
#SBATCH --job-name=nyas_model      ### Job Name
#SBATCH --output=nyas_model.out       ### File in which to store job output
#SBATCH --error=nyas_model.er        ### File in which to store job error messages
#SBATCH --time=0-10:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Number of nodes needed for the job
#SBATCH --ntasks-per-node=1     ### Number of tasks to be launched per Node
#SBATCH --account=bgmp          ### Account used for job submission
#SBATCH --cpus-per-task=1       ##number of cpus (cores) per task
#SBATCH --mem=50000

conda activate bgmp_py37

./nyas_model.py