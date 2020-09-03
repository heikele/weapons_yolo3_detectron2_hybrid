from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
from time import time
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2 import model_zoo
import xml.etree.ElementTree as ET
import PIL.Image as Image
import json
import urllib
from tqdm import tqdm
import pandas as pd
import itertools
import random
import cv2
import numpy as np
import ntpath
import glob
import detectron2
import torchvision
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.init()
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00",
                        "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

weapons_df = pd.read_csv('weapons_anns.csv')


# test bounding boxes
def annotate_image(annotations, resize=True):

    file_name = annotations.file_name.to_numpy()[0]
    print(file_name)
    img = cv2.cvtColor(cv2.imread(
        f'weapons_data/images/train/{file_name}'), cv2.COLOR_BGR2RGB)

    for i, a in annotations.iterrows():
        cv2.rectangle(img, (a.x_min, a.y_min),
                      (a.x_max, a.y_max), (0, 255, 0), 2)

    if not resize:
        return img

    return cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)


# TEST 1
img_df = weapons_df[weapons_df.file_name == weapons_df.file_name.unique()[0]]
img = annotate_image(img_df, resize=False)

plt.imshow(img)
plt.axis('off')
plt.savefig('image_bound_test1.png')

img_df = weapons_df[weapons_df.file_name == weapons_df.file_name.unique()[1]]
img = annotate_image(img_df, resize=False)

plt.imshow(img)
plt.axis('off')
plt.savefig('image_bound_test2.png')

sample_images = [annotate_image(weapons_df[weapons_df.file_name == f])
                 for f in weapons_df.file_name.unique()[:10]]
sample_images = torch.as_tensor(sample_images)
sample_images = sample_images.permute(0, 3, 1, 2)

plt.figure(figsize=(24, 12))
grid_img = torchvision.utils.make_grid(sample_images, nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.savefig('image_bound_test_many.png')

# WEAPON DETECTION

df = pd.read_csv('weapons_anns.csv')
IMAGES_PATH = f'weapons_data'

unique_files = df.file_name.unique()

train_df = pd.read_csv('df_train.csv')
test_df = pd.read_csv('df_val_final.csv')

classes = df.class_name.unique().tolist()


def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):

        record = {}

        image_df = df[df.file_name == img_name]

        file_path = f'{IMAGES_PATH}/images/train/{img_name}'
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)

        objs = []
        for _, row in image_df.iterrows():

            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)

            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(row.class_name),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# Register dataset and metadata catologues
DatasetCatalog._REGISTERED.clear()
#del DatasetCatalog._REGISTERED['weapons_train']
#del DatasetCatalog._REGISTERED['weapons_val']
print(DatasetCatalog._REGISTERED)
print(len(DatasetCatalog._REGISTERED))

entriesToRemove = ('weapons_train', 'weapons_val')
for k in entriesToRemove:
    DatasetCatalog._REGISTERED.pop(k, None)

for d in ["train", "val"]:
    DatasetCatalog.register("weapons_" + d, lambda d=d: create_dataset_dicts(
        train_df if d == "train" else test_df, classes))
    MetadataCatalog.get("weapons_" + d).set(thing_classes=classes)

statement_metadata = MetadataCatalog.get("weapons_train")


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
            # output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()

# The evaluation results will be stored in the `coco_eval` folder if no folder is provided.

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)

cfg.DATASETS.TRAIN = ("weapons_train",)
cfg.DATASETS.TEST = ("weapons_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

predictor = DefaultPredictor(cfg)


evaluator = COCOEvaluator("weapons_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "weapons_val")
inference_on_dataset(trainer.model, val_loader, evaluator)


#experiment_folder = './output/model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'
experiment_folder = './output/'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_metrics = load_json_arr(experiment_folder + 'metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics],
    [x['total_loss'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()

os.makedirs("weapons_results", exist_ok=True)

# THIS MIGHT NEED CHANGING
# with open('./yolov3/data/weap_val_final.txt', 'r') as text:
#    test_image_paths = text.read().split('\n')

test_df = pd.read_csv('df_val_test.csv')

test_image_paths = test_df.file_name.unique()
times = []
data_to_save = []

for clothing_image in test_image_paths:
    t0 = time()
    # print(clothing_image)
    file_path = f'{IMAGES_PATH}/images/train/{clothing_image}'
    im = cv2.imread(file_path)
    outputs = predictor(im)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=statement_metadata,
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')
    # print(instances)
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'weapons_results/{file_name}', result)
    t1 = time()

    times.append(str(t1-t0))
    print(outputs)
with open('detectron2_frameR_test.txt', 'w') as text:
    text.write('\n'.join(times))

#annotated_images = [f'weapons_results/{f}' for f in test_df.file_name.unique()]

#img = cv2.cvtColor(cv2.imread(annotated_images[50]), cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.axis('off')
# plt.savefig('valid_weap.png')
