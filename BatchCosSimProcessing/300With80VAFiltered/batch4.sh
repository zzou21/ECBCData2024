#!/bin/bash
#SBATCH --output=batch4Out.out
#SBATCH --error=batch4Err.err
#SBATCH --mem=20G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing4.py