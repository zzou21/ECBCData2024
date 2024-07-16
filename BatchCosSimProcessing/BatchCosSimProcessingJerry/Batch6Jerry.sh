#!/bin/bash
#SBATCH --output=batch6JerryOut.out
#SBATCH --error=batch6JerryErr.err
#SBATCH --mem=10G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing6.py