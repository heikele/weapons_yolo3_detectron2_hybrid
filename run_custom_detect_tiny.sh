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


#python3 custom_detect.py --cfg yolov3_tiny_custom.cfg --names data/weap.names --source ./data_test/armas\ \(440\).jpg --weights weights/last_weapon_train_yolo_tiny.pt --img 640 --augment
#python3 custom_detect.py --cfg yolov3_tiny_custom.cfg --names data/weap.names --source ./data_test/frame148.jpg --weights weights/last_weapon_train_yolo_tiny.pt --img 640 --augment



python3 custom_detect.py --cfg yolov3_tiny_custom.cfg --names data/weap.names --source ./data_test/ --conf-thres 0.1 --weights weights/last_weapon_train_yolo_tiny.pt --img 640 --augment
