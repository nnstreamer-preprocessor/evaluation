# score txt 파일 로드
# test dataset 과 비교하기
# test dataset은 어떻게 불러오나? //complex yolo 참고
# AP 구해보기

from models import *

import os, sys, time, datetime, argparse

from torch.utils.data import DataLoader

import config as cnf
from kitti_yolo_dataset import KittiYOLODataset

def evaluate(model, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    split='valid'
    dataset = KittiYOLODataset(cnf.root_dir, split=split, mode='EVAL', folder='training', data_aug=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )


f = open("score.txt","r")
scores = f.read()
result = []
print(scores)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")

opt = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initiate model
model = Darknet(opt.model_def).to(device)
# Load checkpoint weights
model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))


print("Compute mAP...")
precision, recall, AP, f1, ap_class = evaluate(
    model,
    iou_thres=opt.iou_thres,
    conf_thres=opt.conf_thres,
    nms_thres=opt.nms_thres,
    img_size=opt.img_size,
    batch_size=opt.batch_size,
)

'''
print("Average Precisions:")
for i, c in enumerate(ap_class):
    print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

print(f"mAP: {AP.mean()}")
'''