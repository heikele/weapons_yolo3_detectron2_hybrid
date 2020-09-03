#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1               
#SBATCH --account=project_2000859

module purge
module load python-data
module load gcc/8.3.0
module load cuda/10.1.168
module load cudnn/7.6.1.34-10.1


date;hostname;pwd


python3 detect_detectron.py