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
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts, DatasetFromList, MapDataset, DatasetMapper
from detectron2.data.samplers import InferenceSampler

from lib.visualizer import Visualizer
from lib.keypointtrainer import KeypointTrainer
from lib.evaluator import CobraCOCOEvaluator
from lib.config import add_freeze_config, add_keypoint_alphas

def parse_args():
	parser = argparse.ArgumentParser(description='Train Hand Bounding Box and Keypoint Detector')
	parser.add_argument('--cfg', type=str, required=True)
	return parser.parse_args()


def register_metadata(dataset):
	MetadataCatalog.get(dataset).set(keypoint_names=[str(i) for i in range(1, 22)])

	flip_map = {'1': '1', '2': '2', '3': '3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12': '12', 
	'13': '13', '14': '14', '15':'15', '16':'16', '17':'17', '18': '18', '19':'19', '20':'20', '21':'21'}

	MetadataCatalog.get(dataset).set(keypoint_flip_map=flip_map, keypoint_pck_params={})


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
	elif cfg.DATASETS.TRAIN[0] == "train_2_fingers":
		register_coco_instances("train_2_fingers", {}, "./annotations/train_two_fingers.json", "./images/all_images")
		register_coco_instances("val_2_fingers", {}, "./annotations/val_two_fingers.json", "./images/all_images")
	elif cfg.DATASETS.TRAIN[0] == "train_man" and len(cfg.DATASETS.TRAIN) > 0:
		register_coco_instances("train_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_train.json", "./images/CMU")
		register_coco_instances("val_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_val.json", "./images/CMU")
		register_coco_instances("test_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_test.json", "./images/CMU")
		register_coco_instances("train_pan", {}, "./annotations/CMU_annotations/hand_labels_panoptic/v0.0.2/annotation_train.json", "./images/CMU")
		register_coco_instances("val_pan", {}, "./annotations/CMU_annotations/hand_labels_panoptic/v0.0.2/annotation_val.json", "./images/CMU")
		register_coco_instances("train_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_train.json", "./images/CMU")
		register_coco_instances("val_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_val.json", "./images/CMU")
		register_coco_instances("test_synth", {}, "./annotations/CMU_annotations/hand_labels_synth/v0.0.1/annotation_test.json", "./images/CMU")
	elif cfg.DATASETS.TRAIN[0] == "train_man":
		register_coco_instances("train_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_train.json", "./images/CMU")
		register_coco_instances("val_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_val.json", "./images/CMU")
		register_coco_instances("test_man", {}, "./annotations/CMU_annotations/hand_labels/v0.0.2/annotation_test.json", "./images/CMU")		

	# ERROR
	else:
		print("ERROR: Could not register datasets described in provided configuration file.")
		exit(0)

def build_detection_data_loader(cfg, dataset, mapper=None):
	dataset_dicts = get_detection_dataset_dicts(
		dataset,
		filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
		min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
		if cfg.MODEL.KEYPOINT_ON
		else 0,
		proposal_files=[
		cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
		] if cfg.MODEL.LOAD_PROPOSALS else None,
	)
	dataset = DatasetFromList(dataset_dicts)
	if mapper is None:
		mapper = DatasetMapper(cfg, False)
	dataset = MapDataset(dataset, mapper)

	sampler = InferenceSampler(len(dataset))
	# Always use 1 image per worker during inference since this is the
	# standard when reporting inference time in papers.
	batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

	data_loader = torch.utils.data.DataLoader(
		dataset,
		num_workers=cfg.DATALOADER.NUM_WORKERS,
		batch_sampler=batch_sampler,
		collate_fn=self.trivial_batch_collator,
	)
	return data_loader

def train():
	args = parse_args()

	cfg = get_cfg()
	add_freeze_config(cfg)
	add_keypoint_alphas(cfg)
	cfg.merge_from_file(args.cfg)
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	register_data(cfg)

	for dataset in cfg.DATASETS.TRAIN:
		register_metadata(dataset)

	for dataset in cfg.DATASETS.TEST:
		register_metadata(dataset)


	print("Training...")
	trainer = KeypointTrainer(cfg) 

	if cfg.MODEL.FREEZE:
		for name, p in trainer.model.named_parameters():
			if "keypoint" not in name:
				p.requires_grad = False

	trainer.resume_or_load(resume=False)
	trainer.train()

	print("Evaluating...")

	evaluator = CobraCOCOEvaluator("val", cfg, False, output_dir=cfg.OUTPUT_DIR)
	val_loader = build_detection_test_loader(cfg, "val")
	inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
	train()