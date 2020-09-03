#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1               
#SBATCH --account=project_2000859

module purge
module load python-data
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.6.1.34-10.1


date;hostname;pwd

#export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-10.0/targets/x86_64-linux/lib/ 

python3 train.py --data coco64.data --batch 16 --accum 1 --epochs 300 --nosave --cache --weights '' --name from_scratch
python3 train.py --data coco64.data --batch 16 --accum 1 --epochs 300 --nosave --cache --weights yolov3-spp-ultralytics.pt --name from_yolov3-spp-ultralytics
python3 train.py --data coco64.data --batch 16 --accum 1 --epochs 300 --nosave --cache --weights darknet53.conv.74 --name from_darknet53.conv.74
python3 train.py --data coco1.data --batch 1 --accum 1 --epochs 300 --nosave --cache --weights darknet53.conv.74 --name 1img
python3 train.py --data coco1cls.data --batch 16 --accum 1 --epochs 300 --nosave --cache --weights darknet53.conv.74 --cfg yolov3-spp-1cls.cfg --name 1cls