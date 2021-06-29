import os
import re
import shutil 
import json
import uuid

# open the jsons
hand_train = json.load(open("./CMUPanoptic/annotations/hand_labels/v0.0.2/annotation_train.json"))
hand_test = json.load(open("./CMUPanoptic/annotations/hand_labels/v0.0.2/annotation_test.json"))
hand_val = json.load(open("./CMUPanoptic/annotations/hand_labels/v0.0.2/annotation_val.json"))

pano_train = json.load(open("./CMUPanoptic/annotations/hand_labels_panoptic/v0.0.2/annotation_train.json"))
pano_val = json.load(open("./CMUPanoptic/annotations/hand_labels_panoptic/v0.0.2/annotation_val.json"))

synth_train = json.load(open("./CMUPanoptic/annotations/hand_labels_synth/v0.0.1/annotation_train.json"))
synth_test = json.load(open("./CMUPanoptic/annotations/hand_labels_synth/v0.0.1/annotation_test.json"))
synth_val = json.load(open("./CMUPanoptic/annotations/hand_labels_synth/v0.0.1/annotation_val.json"))

# combine the coco
# relable image IDs, annotation IDs
val_paths = set() # "hand143_panopticdb/..."
train_paths = set()

train = {}
val = {}
train['categories'] = hand_train['categories']
val['categories'] = hand_train['categories']
train['images'] = []
val['images'] = []
train['annotations'] = []
val['annotations'] = []

# go through train
image_id = 1
annotation_id = 1

for json_blob in [hand_train, hand_val, pano_train, synth_train, synth_val]:
	new_to_old = {}

	for img in json_blob["images"]:
		train_paths.add(img["file_name"])
		new_to_old[img["id"]] = image_id
		img["id"] = image_id

		image_id += 1

		train['images'].append(img)

	for ann in json_blob["annotations"]:
		ann["image_id"] = new_to_old[ann["image_id"]]
		ann["id"] = annotation_id

		annotation_id += 1

		train['annotations'].append(ann)

# go through val
for json_blob in [hand_test, pano_val, synth_test]:
	new_to_old = {}

	for img in json_blob["images"]:
		val_paths.add(img["file_name"])
		new_to_old[img["id"]] = image_id
		img["id"] = image_id

		image_id += 1

		val['images'].append(img)

	for ann in json_blob["annotations"]:
		ann["image_id"] = new_to_old[ann["image_id"]]
		ann["id"] = annotation_id

		annotation_id += 1

		val['annotations'].append(ann)


names = set()
new_names = {} 
# move the images
for directory in ["./CMUPanoptic/hand_labels/manual_train", "./CMUPanoptic/hand_labels/manual_test", "./CMUPanoptic/hand143_panopticdb/imgs", "./CMUPanoptic/hand_labels_synth/synth1", "./CMUPanoptic/hand_labels_synth/synth2", "./CMUPanoptic/hand_labels_synth/synth3", "./CMUPanoptic/hand_labels_synth/synth4"]:
	for entry in os.scandir(directory):
		if entry.path.endswith(".jpg"):
			# print(entry.path)
			fname = os.path.split(entry.path)[1]
			if fname not in names:
				names.add(fname)
			else:
				fname = str(uuid.uuid4()) + fname
				new_names[str(entry.path)[14:]] = os.path.join(os.path.split(entry.path)[0][14:], fname)

			# print(fname)
			if entry.path[14:] in val_paths:
				# move to val
				shutil.move(entry.path, "./data/coco/images/CMU_test/" + fname)
			elif entry.path[14:] in train_paths:
				# move to train
				shutil.move(entry.path, "./data/coco/images/CMU_train/" + fname)

# input new names into coco
for img in train["images"]:
	if img["file_name"] in new_names:
		img["file_name"] = new_names[img["file_name"]]

for img in val["images"]:
	if img["file_name"] in new_names:
		img["file_name"] = new_names[img["file_name"]]

# save new coco
json.dump(train, open("./data/coco/annotations/hand_keypoints_CMU_train.json", "w"))
json.dump(val, open("./data/coco/annotations/hand_keypoints_CMU_test.json", "w"))
