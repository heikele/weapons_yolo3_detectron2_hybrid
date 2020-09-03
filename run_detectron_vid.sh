#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1               
#SBATCH --account=project_2000859

date;hostname;pwd

python3 vid_detectron.py