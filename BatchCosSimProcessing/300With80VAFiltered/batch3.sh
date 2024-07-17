#!/bin/bash
#SBATCH --output=batch3Out.out
#SBATCH --error=batch3Err.err
#SBATCH --mem=20G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing3.py