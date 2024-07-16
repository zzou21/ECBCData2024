#!/bin/bash
#SBATCH --output=batch5JerryOut.out
#SBATCH --error=batch5JerryErr.err
#SBATCH --mem=10G
#SBATCH --partition=gpu-common

srun --cpu-bind=None python ClusterUsingCosineSimilarityForBatchProcessing5.py