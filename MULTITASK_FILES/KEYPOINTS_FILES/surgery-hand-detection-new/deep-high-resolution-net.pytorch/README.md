# Surgery Hand Keypoint Detection Model

Code for implementing hand keypoint detection on the Youtube Surgery dataset. The final model selected from these experiments (`keypoints_final.pth`) predicts 21 keypoints corresponding to the joints in a hand and the wrist. Our final model has a PCK@0.5 of 0.493 and 0.40 on the Youtube Surgery Hands dataset validation and test splits, respectively.

#### Contact
For questions about this repo, contact Krishna at kkpatel at cs dot stanford dot edu. 

## Documentation
See more detailed notes regarding these experiments [here](https://docs.google.com/document/d/1zC7RxGLEi-Pg1D3CvS2gJ5m-IAwdt7Apc6cQPj23Tj8/edit?usp=sharing). 

Additionally, extra information about the original model architecture can be found [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

## Setup

### GPU Specifications

Aquire a GPU instance with CUDA 10.1 by either requesting GPUs through [Stanford Clusters](https://github.com/yeung-lab/marvl-clusters-guide) or launching an AMI with CUDA 10.1.

### Conda Environment
Install required dependencies.

```
conda env create --file environment.yml
source activate keypoints

cd lib
make
cd ..
```

#### Install [COCOAPI](https://github.com/cocodataset/cocoapi)

```
git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI
python3 setup.py install --user
```

### Download Models 
Ensure that you are in the `deep-high-resolution-nets.pytorch` directory. 

```
mkdir models
mkdir models/imagenet
mkdir models/pose_hrnet
cd models
```

Either download the models folder from [Google Drive](https://drive.google.com/file/d/1Tk8dXtLPKxnIbf8FmCm9gqrzXtmpVNDa/view?usp=sharing) and unzip it in the `deep-high-resolution-nets.pytorch/models` directory, or use the following to download the weights from SAIL.

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/trained.zip ./
unzip trained.zip
```

Depending on the experiments that you are running, if necessary, you should also download relevant models from the [original paper](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and place them into the `models/imagenet` and `models/pose_hrnet` folders depending on the models you choose.


#### Models Included
See documentation [here](https://docs.google.com/document/d/1zC7RxGLEi-Pg1D3CvS2gJ5m-IAwdt7Apc6cQPj23Tj8/edit?usp=sharing) for more information on each model.

* `keypoints_final.pth`: Final model selected (DHRN-48)
* `cmu_48_best.pth`: Deep High Resolution Net - 48 pretrained on CMU Panoptic Hand Dataset, performance of PCK@0.1 0.196 on CMU test sets.
* `cmu_32_best.pth`: Deep High Resolution Net - 32 pretrained on CMU Panoptic Hand Dataset, performance of PCK@0.1 0.177 on CMU test sets.
* `overfit_surgery_32.pth`: Deep High Resolution Net - 32 pretrained on CMU Panoptic Hand Dataset, overfit on Surgery Hands dataset, train performance of PCK@0.1 0.25, used to bootstrap annotations
* `keypoints_32.pth`: Deep High Resolution Net - 32 pretrained on CMU Panoptic Hand Dataset, trained on Surgery Hands dataset, performance of PCK@0.5 0.489


### Unzip Annotations
Unzip annotations that must be compressed to fit on GitHub.

```
cd data/coco/annotations
unzip hand_keypoints_train_boot_aug_2.json.zip
```

### Download Datasets
Download the necessary datasets for running the experiments you are replicating. Place these datasets into a separate `images` directory under `data/coco`.

```
cd data/coco
mkdir images
cd images
```

#### Youtube Surgery Hands Dataset
You can download the Youtube Surgery Hands Dataset [train](https://drive.google.com/file/d/1vd3mmY6WFo6Ih5Yk3sjBvA3SRB1UDwWU/view?usp=sharing) and [validation](https://drive.google.com/file/d/1vd3mmY6WFo6Ih5Yk3sjBvA3SRB1UDwWU/view?usp=sharing) splits from Google Drive and unzip it in the `images` directory, or use the following to download the data from SAIL.

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/train_keypoints.zip ./
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/val_keypoints.zip ./
unzip train_keypoints.zip
unzip val_keypoints.zip
```

#### Youtube Surgery Hands Bootstrapped Annotations 
Download these images from SAIL:

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/bootstrap.zip ./
unzip bootstrap.zip
mv bootstrap train_boot
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
To utilize CMU Panoptic Hand data in this framework, we must first do some preprocessing. First, move up directories to the `deep-high-resolution-nets.pytorch` directory.

```
scp -r [CSID]@sc.stanford.edu:/pasteur/u/arjundd/data/CMUPanoptic ./

cd CMUPanoptic
unzip hand_labels_synth.zip
unzip hand_labels.zip
tar -xf hand143_panopticdb.tar
cd ..

python3 filter_images.py
```

## Hand Keypoint Detection

### Training a Model 
Train a model by supplying a configuration file to the `tools/train.py` script. For example, to train a model with the same configuration as our final model, use the following command.

```
python3 tools/train.py --cfg hand_exp/cmu_surg_boot_final.yaml
```

NOTE: If you are training a model utilzing CMU data, please uncomment lines 136-7 [here](https://github.com/yeung-lab/surgery-hand-detection/blob/168e0e5b78b00eb3c2471fc1f6fd8e2e30ba8dea/deep-high-resolution-net.pytorch/lib/dataset/coco.py#L135) and comment out line 140.

### Evaluating a Model (Legacy)
Evaluate a model by supplying a configuration file to the `tools/test.py` script. For example, to evaluate our final model on the test set, using the Deep High Resolution Nets framework the following command.

```
python3 tools/test.py --cfg hand_exp/cmu_surg_boot_final.yaml TEST.MODEL_FILE models/trained/keypoints_final.pth
```

### Evaluating a Model (Accurate)
Evaluate a model by following the directions to use the `PCKh.py` [script](https://github.com/yeung-lab/surgery-hand-detection/blob/master/keypoint-analysis/PCKh.py) in `../keypoint-analysis`.

### Visualization and Analysis
See surgery hand keypoint visualization and analysis in `../keypoint-analysis` [here](https://github.com/yeung-lab/surgery-hand-detection/tree/master/keypoint-analysis).
