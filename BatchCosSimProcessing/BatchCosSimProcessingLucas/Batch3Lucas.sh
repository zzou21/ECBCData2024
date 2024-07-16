#!/bin/bash
#SBATCH --output=batch3LucasOut.out
#SBATCH --error=batch4LucasOut.err
#SBATCH --mem=10G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing3.py