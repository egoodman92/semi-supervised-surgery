import torch, torchvision

import detectron2
import argparse
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from lib.visualizer import Visualizer

from lib.customtrainer import CustomTrainer
from lib.augtrainer import AugTrainer, AugTrainerLight
from lib.config import add_aug_config

def parse_args():
	parser = argparse.ArgumentParser(description='Train Hand Bounding Box Detector')
	parser.add_argument('--cfg', type=str, required=True)
	return parser.parse_args()


def register_data(cfg):
	print("Registering data ...")
	if cfg.DATASETS.TRAIN[0] == "train_original_anns":
		register_coco_instances("train_original_anns", {}, "./annotations/train_5_10.json", "./images/all_images")
		register_coco_instances("val_original_anns", {}, "./annotations/val_5_10.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_validated":
		register_coco_instances("train_validated", {}, "./annotations/train_validated.json", "./images/all_images")
		register_coco_instances("val_validated", {}, "./annotations/val_validated.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_boot_aug":
		register_coco_instances("train_boot_aug", {}, "./annotations/train_boot_aug_2.json", "./images/train_boot_aug_2")
		register_coco_instances("val_validated", {}, "./annotations/val_validated.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_aug":
		register_coco_instances("train_aug", {}, "./annotations/train_aug_823.json", "./images/train_aug_823")
		register_coco_instances("val_validated", {}, "./annotations/val_validated.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_chirality":
		register_coco_instances("train_chirality", {}, "./annotations/train_chirality.json", "./images/all_images")
		register_coco_instances("val_chirality", {}, "./annotations/val_chirality.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_multi":
		register_coco_instances("train_multi", {}, "./annotations/train_multi.json", "./images/all_images")
		register_coco_instances("val_multi", {}, "./annotations/val_multi.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_egohands":
		register_coco_instances("train_egohands", {}, "./annotations/train_egohands.json", "./images/egohands_data/_LABELLED_SAMPLES")
		register_coco_instances("test_egohands", {}, "./annotations/test_egohands.json", "./images/egohands_data/_LABELLED_SAMPLES")
	elif cfg.DATASETS.TRAIN[0] == "train_oxford":
		register_coco_instances("train_oxford", {}, "./annotations/train_oxford.json", "./images/oxford_hands")
		register_coco_instances("test_oxford", {}, "./annotations/test_oxford.json", "./images/oxford_hands")
	elif cfg.DATASETS.TRAIN[0] == "train_man":
		register_coco_instances("train_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_train.json", "./images/CMU")
		register_coco_instances("val_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_val.json", "./images/CMU")
		register_coco_instances("test_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_test.json", "./images/CMU")
		register_coco_instances("train_pan", {}, "./annotations/CMU_annotations/hand_labels_panoptic/v0.0.2/annotation_train.json", "./images/CMU")
		register_coco_instances("val_pan", {}, "./annotations/CMU_annotations/hand_labels_panoptic/v0.0.2/annotation_val.json", "./images/CMU")
		register_coco_instances("train_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_train.json", "./images/CMU")
		register_coco_instances("val_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_val.json", "./images/CMU")
		register_coco_instances("test_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_test.json", "./images/CMU")

	# ERROR
	else:
		print("ERROR: Could not register datasets described in provided configuration file.")
		exit(0)

def train():
	args = parse_args()

	cfg = get_cfg()
	add_aug_config(cfg)
	cfg.merge_from_file(args.cfg)
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

	register_data(cfg)

	print("Training...")
	if cfg.DATALOADER.AUG == "Light":
		trainer = AugTrainerLight(cfg)
	elif cfg.DATALOADER.AUG == "Normal":
		trainer = AugTrainer(cfg)
	else:
		trainer = CustomTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()

	print("Evaluating...")

	evaluator = COCOEvaluator("val", cfg, False, output_dir=cfg.OUTPUT_DIR)
	val_loader = build_detection_test_loader(cfg, "val")
	inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
	train()