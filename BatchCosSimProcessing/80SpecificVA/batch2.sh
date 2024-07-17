#!/bin/bash
#SBATCH --output=batch2Out.out
#SBATCH --error=batch2Err.err
#SBATCH --mem=20G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing2.py