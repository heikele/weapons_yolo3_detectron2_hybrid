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
module load cuda/10.0.130
module load cudnn/7.6.1.34-10.1


date;hostname;pwd

python3 detect.py --cfg yolov3_custom_weap.cfg --names data/weap.names --source pistol_video_short.mp4  --weights weights/last_weapon_yolo_full.pt 