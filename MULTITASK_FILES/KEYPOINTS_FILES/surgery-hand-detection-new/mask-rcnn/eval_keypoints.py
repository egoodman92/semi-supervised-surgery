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
from detectron2.evaluation import inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts, DatasetFromList, MapDataset, DatasetMapper
from detectron2.data.samplers import InferenceSampler


from lib.visualizer import Visualizer
from lib.evaluator import CobraCOCOEvaluator
from lib.config import add_freeze_config, add_keypoint_alphas


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate Hand Bounding Box and Keypoint Detector')
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


def evaluate():
	args = parse_args()

	cfg = get_cfg()
	add_freeze_config(cfg)
	add_keypoint_alphas(cfg)
	cfg.merge_from_file(args.cfg)
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

	register_coco_instances("test", {}, "./annotations/test_validated.json", "./images/all_images")
	MetadataCatalog.get("test").set(keypoint_names=[str(i) for i in range(1, 22)])

	flip_map = {'1': '1', '2': '2', '3': '3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12': '12', 
	'13': '13', '14': '14', '15':'15', '16':'16', '17':'17', '18': '18', '19':'19', '20':'20', '21':'21'}

	MetadataCatalog.get("test").set(keypoint_flip_map=flip_map, keypoint_pck_params={})
	predictor = DefaultPredictor(cfg)

	if args.save_imgs:
		visualize(cfg, predictor)
	
	print("Evaluating...")
	evaluator = CobraCOCOEvaluator("test", cfg, False, output_dir=cfg.OUTPUT_DIR)
	test_loader = build_detection_data_loader(cfg, "test", DatasetMapper(cfg, True))
	inference_on_dataset(predictor.model, test_loader, evaluator)


if __name__ == '__main__':
	evaluate()
