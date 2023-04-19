#!/bin/bash
# Script to help launch SLURM jobs from another script
cd /scratch/heminway.r/LearningInTheGym/slurm/
source activate EAResearch
python gaTrainOneFile.py $1