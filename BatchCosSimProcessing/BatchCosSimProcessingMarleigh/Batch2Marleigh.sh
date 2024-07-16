#!/bin/bash
#SBATCH --output=batch2MarleighOut.out
#SBATCH --error=batch2MarleighErr.err
#SBATCH --mem=10G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing2.py