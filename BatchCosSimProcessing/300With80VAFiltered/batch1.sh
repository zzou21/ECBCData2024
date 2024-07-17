#!/bin/bash
#SBATCH --output=batch1Out.out
#SBATCH --error=batch1Err.err
#SBATCH --mem=20G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing1.py