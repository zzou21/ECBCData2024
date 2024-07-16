#!/bin/bash
#SBATCH --output=batch1MarleighOut.out
#SBATCH --error=batch1MarleighErr.err
#SBATCH --mem=10G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing1.py