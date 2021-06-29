# Surgery Hand and Keypoint Detection Models

Code for implementing hand bounding box detection and hand keypoint detection on the Youtube Surgery dataset. While this repo contains code for implementing both hand bounding box detection and keypoint detection, the final model used from these experiments (`bbox_final.pth`) only predicts bounding boxes. Our final model has a mAP of 80.942 and 82.388 on the Youtube Surgery Hands dataset validation and test splits, respectively.

#### Contact
For questions about this repo, contact Krishna at kkpatel at cs dot stanford dot edu.

## Documentation
See more detailed notes regarding these experiments [here](https://docs.google.com/document/d/1zC7RxGLEi-Pg1D3CvS2gJ5m-IAwdt7Apc6cQPj23Tj8/edit?usp=sharing).

Additionally, documentation regarding this architecture can be found [here](https://detectron2.readthedocs.org).

## Setup

### GPU Specifications

Aquire a GPU instance with CUDA 10.1 by either requesting GPUs through [Stanford Clusters](https://github.com/yeung-lab/marvl-clusters-guide) or launching an AMI with CUDA 10.1.

### Conda Environment
Install required dependencies.

```
conda env create --file environment.yml
source activate surgery
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

### Download Models 
Ensure that you are in the `mask-rcnn` directory. Either download the models folder from [Google Drive](https://drive.google.com/file/d/1UDDYMqX2JJPPILlvmxxhTlVo4cDGUshg/view?usp=sharing) and unzip it in the `mask-rcnn` directory, or use the following to download the weights from SAIL.

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/models.zip ./
unzip models.zip
```

#### Models Included
See documentation [here](https://docs.google.com/document/d/1zC7RxGLEi-Pg1D3CvS2gJ5m-IAwdt7Apc6cQPj23Tj8/edit?usp=sharing) for more information on each model.

* `bbox_final.pth`: Final model selected
* `bbox_and_keypoints.pth`: Best Keypoint-RCNN, pretrained on COCO, fine-tuned on Surgery Hands bootstrapped annotations with 20:1 data augmention, mAP \~68, PCK@0.5 \~0.457

* `egohands_final.pth`: Mask-RCNN pretrained on COCO, fine-tuned on EgoHands (capable of segmentation and bounding box predictions).
* `oxford_final.pth`:Faster-RCNN pretrained on COCO, fine-tuned on Oxford Hands.

* `egohands_oxford_final.pth`: Faster-RCNN pretrained on COCO, Egohands, fine-tuned on Oxford Hands.
* `oxford_egohands_final.pth`: Mask-RCNN pretrained on COCO, Oxford Hands, fine-tuned on EgoHands (capable of segmentation and bounding box predictions).
* `egohands_surg_final.pth`: Faster-RCNN pretrained on COCO, Egohands, fine-tuned on original surgery dataset annotations.

* `cmu_final.pth`: Keypoint-RCNN pretrained on COCO, fine-tuned on full CMU Panoptic Hand Dataset (capable of bounding box and keypoint predictions).
* `cmu_manual.pth`: Keypoint-RCNN pretrained on COCO, fine-tuned on manually labeled data from the CMU Panoptic Hand Dataset (capable of bounding box and keypoint predictions).


### Unzip Annotations
Unzip annotations that must be compressed to fit on GitHub.

```
cd annotations
unzip train_boot_aug_2.json.zip
```

### Download Datasets
Download the necessary datasets for running the experiments you are replicating. Place these datasets into a separate `images` directory.

```
mkdir images
cd images
```

#### Youtube Surgery Hands Dataset
You can download the Youtube Surgery Hands Dataset from [Google Drive](https://drive.google.com/file/d/1AwOHVGvoEe3iT9nEAJP24p6sVReATccM/view?usp=sharing) and unzip it in the `images` directory, or use the following to download the data from SAIL.

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/all_images.zip ./
unzip all_images.zip
```

#### EgoHands Dataset
Download these images from SAIL:

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/egohands_data.zip ./
unzip egohands_data.zip
```

#### Oxford Hands Dataset
Download these images from SAIL:

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/oxford_hands.zip ./
unzip oxford_hands.zip
```

#### Youtube Surgery Hands 20:1 Kitchen Sink Data Augmentation
Download these images from SAIL:

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/train_aug_823.zip ./
unzip train_aug_823.zip
```

#### Youtube Surgery Hands Bootstrapped Annotations + 20:1 Kitchen Sink Data Augmentation
Download these images from SAIL:

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/train_boot_aug_2.zip ./
unzip train_boot_aug_2.zip
```

#### CMU Panoptic Hands
Download these images from SAIL:

```
mkdir CMU
cd CMU
scp [CSID]@sc.stanford.edu:/pasteur/u/arjundd/data/CMUPanoptic/hand143_panopticdb.tar ./
scp [CSID]@sc.stanford.edu:/pasteur/u/arjundd/data/CMUPanoptic/hand_labels_synth.zip ./
scp [CSID]@sc.stanford.edu:/pasteur/u/arjundd/data/CMUPanoptic/hand_labels.zip ./
unzip hand_labels_synth.zip
unzip hand_labels.zip
tar -xf hand143_panopticdb.tar
```

## Bounding Box Detection

### Training a Model 
Train a model by supplying a configuration file to the `train_bbox.py` script. For example, to train a model with the same configuration as our final model, use the following command.

```
python3 train_bbox.py --cfg ./experiments/bbox/coco_ego_surg_val.yaml
```

### Evaluating a Model
Evaluate a model by supplying a configuration file to the `eval_bbox.py` script. For example, to evaluate our final model on the test set, use the following command.

```
python3 eval_bbox.py --cfg ./experiments/bbox/final_model.yaml
```

#### Visualizing Evaluations
To visualize a sample of a model's performance on the test split, use the `--save_imgs` flag:

```
python3 eval_bbox.py --cfg ./experiments/bbox/final_model.yaml --save_imgs 
```

## Bounding Box and Keypoint Detection

### Training a Model 
Train a model by supplying a configuration file to the `train_keypoints.py` script:

```
python3 train_keypoints.py --cfg ./experiments/keypoints/coco_surg.yaml
```

### Evaluating a Model
Evaluate a model by supplying a configuration file to the `eval_bbox.py` script. For example, to evaluate our final model on the test set, use the following command.

```
python3 eval_keypoints.py --cfg ./experiments/keypoints/coco_surg.yaml
```

#### Visualizing Evaluations
To visualize a sample of a model's performance on the test split, use the `--save_imgs` flag:

```
python3 eval_keypoints.py --cfg ./experiments/keypoints/coco_surg.yaml --save_imgs
```
