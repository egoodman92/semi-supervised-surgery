import torch, torchvision

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os
import random

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

from lib.visualizer import Visualizer
from lib.config import add_aug_config

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate Hand Bounding Box Detector')
	parser.add_argument('--cfg', type=str, required=True)
	parser.add_argument('--save_imgs', action='store_true')
	return parser.parse_args()


def visualize(cfg, predictor):
	for i, d in enumerate(DatasetCatalog.get("test")):    
		im = cv2.imread(d["file_name"])
		outputs = predictor(im)
		v = Visualizer(im[:, :, ::-1],
					   metadata=MetadataCatalog.get("test"))
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		cv2.imwrite(cfg.OUTPUT_DIR + "/predicted_" + str(i) + ".png", v.get_image()[:, :, ::-1])


def evaluate():
	args = parse_args()

	cfg = get_cfg()
	add_aug_config(cfg)
	cfg.merge_from_file(args.cfg)
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

	register_coco_instances("test", {}, "./annotations/test_validated.json", "./images/all_images")
	predictor = DefaultPredictor(cfg)

	if args.save_imgs:
		visualize(cfg, predictor)
	
	print("Evaluating...")
	evaluator = COCOEvaluator("test", cfg, False, output_dir=cfg.OUTPUT_DIR)
	test_loader = build_detection_test_loader(cfg, "test")
	inference_on_dataset(predictor.model, test_loader, evaluator)


if __name__ == '__main__':
	evaluate()
