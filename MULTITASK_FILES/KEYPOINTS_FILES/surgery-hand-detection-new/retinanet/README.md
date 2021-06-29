# Surgery Hands Detection Model

Code for implementing hand detection on the updated surgery hands dataset. 

Borrows from the [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet) implementation.

For code on how the updated dataset was constructed (using data available on the Yeung Lab drive), see this [CoLab notebook](https://colab.research.google.com/drive/1WSXqQEiZZOHU9d-0ss_eI43FDoo8HDFB).

## Setup

Initiated the following conda environment:
```
conda create -n cv python=3.6 
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install pandas
conda install requests
conda install opencv (pip install opencv-python)
conda install -c conda-forge pycocotools (pip install pycocotools)
conda install -c conda-forge scikit-image
```

Note: The `pytorch-retinanet` project has its own README included that has instructions for installing dependencies. Setting up the virtual environment as above should take care of that (i.e. don't need to follow them and re-install things), so you should be good to go.

### Pastuer Interactive GPU  
Example commands to request GPU (NVIDIA RTX 2080 Ti GPUs with 11GB of GPU memory):
```
srun -N 1 -c 1 --time=00:25:00 --mem=25600M -p pasteur --nodelist=pasteur[1] --gres=gpu:1 --pty bash
srun -N 1 -c 1 --time=00:25:00 --mem=25600M -p pasteur -c 1 --mem-per-cpu=64g --nodelist=pasteur[1] --gres=gpu:1 --pty bash
```

Or run this to request other GPU (NVIDIA RTX Titan X GPUs with 24GB of GPU memory):
```
srun -N 1 -c 1 --time=02:00:00 --mem=25600M -p pasteur --nodelist=pasteur[2] --gres=gpu:1 --pty bash
srun -N 1 -c 1 --time=00:25:00 --mem=50000M -p pasteur -c 2 --mem-per-cpu=16gb --nodelist=pasteur[2] --gres=gpu:1 --pty bash
srun -N 1 -c 1 --time=00:30:00 --mem=25600M -p pasteur --nodelist=pasteur[2] --gres=gpu:1 --pty bash
```

Then activate the venv with `conda activate cv`

## Directory Organization
* `pytorch-retinanet`: code for the RetinaNet model as well preparing data, training + evaluating the model  
* `hand-detection-dataset`: contains current surgery hands dataset (images + annotations), and supporting scripts  
* `oxford-hand-detection-dataset`: Oxford hands dataset (images + annotations)  
* `old-surgery-hand-detection-dataset`: surgery hands dataset with previous (incorrect) train / val / test splits. Frames are specific to a split, but the videos they come from may be represented in multiple splits.

## Dataset 
We use the csv format for annotations coupled with surgery frames split into train, validation, and test directories. Currently the updated surgery hands dataset is available at:  
* SAIL: `/pasteur/u/mzhang/surgery-hands/hand-detection-dataset` 

Note that images (for the surgery hands dataset and other datasets) should be copied directly from their respective `train` / `val` / `test`  directories in `/pasteur/u/mzhang/surgery-hands/`.

### Annotation Descriptions
1. `annotation_[split name]-by_vid.csv` - Annotations of each video, shuffled by video, also shuffled by row  
2. `annotation_[split name]-by_vid-unshuffled.csv` - Same as above, but not shuffled by row  
3. `annotation_[split name].csv` - Same as (1), but does not include header names or the `video` column. Use this for the RetinaNet CSV dataloader.

## Quick Data Setup
To make the data more accessible, first `cd` to the `pytorch-retinanet` directory and run `process_data.py`, which will copy and edit the annotation data from its source directory to the working one.  
* You should check the script first to make sure the data variables are set to what you want.
***Note on data setup*** - Out of convention, the actual images are split into separate train, val, test folders, and these are reflected in the path names. It may or may not be more convenient to consolidate the images to a single directory, but make sure to update the paths in the annotation files accordingly.

## Model Training
To train, we use the code in [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet). More specifically we can run:
```
python train.py --dataset csv --csv_train annotation_train.csv --csv_classes ../hand-detection-dataset/class_list.csv --depth 50 --csv_val annotation_val.csv
```
from within `pytorch-retinanet` to train with a ResNet50 backbone architecture.

Example sample run with more parameters:  
```
python train.py --dataset csv --csv_train annotation_train.csv --csv_classes ../hand-detection-dataset/class_list.csv --depth 50 --csv_val annotation_val.csv --learning_rate 1e-4 --batch_size 16
```

Training on updated COCO data
```
python train.py --dataset coco --coco_path ./data_coco --depth 50
python train.py --dataset coco --coco_path /pasteur/data/YoutubeSurgery/annotations_hands/v0.1.0/ --depth 50
```

Training on Oxford Hands (Surgery Hands validation):
```
python train.py --dataset csv --csv_train oxford-annotation_train.csv --csv_classes ../hand-detection-dataset/class_list.csv --depth 50 --model_name oxford --csv_val annotation_val.csv --learning_rate 1e-4 --batch_size 16
```

## Model Evaluation  
To evaluate a model, we run `evaluate.py`, which can take in either the validation or testing dataset for the `--csv_val` argument. As an example using the pre-trained COCO model downloaded from [here](https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS), we can run:
```
python evaluate.py --csv_val annotation_val.csv --csv_classes ../hand-detection-dataset/class_list.csv --model_path ./models/coco_resnet_50_map_0_335_state_dict.pt
```

Or to test a model we trained on our own:
```
python evaluate.py --csv_val annotation_test.csv --csv_classes ../hand-detection-dataset/class_list.csv --model_path ./models/csv_retinanet-e=42-lr=0.0001-bs=16-npt=False.pt
```
On the updated coco data, we use:
```
python evaluate.py --set_name_test annotation_test --coco_path /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/data_coco/ --model_path /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/models/coco_retinanet-surg_coco_tv-final-lr=1e-05-bs=2-npt=False.pt
```
On the tracking test images, use:
```
python evaluate.py --csv_val annotation_test_tracking.csv --csv_classes ../hand-detection-dataset/class_list.csv --model_path ./models/coco_retinanet-ego_oxford_surg-e=9-lr=1e-05-bs=4-npt=False.pt
```

Evaluation with oxford trained:
```
python evaluate.py --set_name_test annotation_test --coco_path /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/data_coco/ --model_path /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/models/old_csv/csv_retinanet-oxford_train_val-final-lr=1e-05-bs=4-npt=False.pt
```


## Visualization
To visualize our results, we can use `visualize.py`, e.g.
```
python visualize.py --dataset csv --csv_classes ../hand-detection-dataset/class_list.csv  --csv_val annotation_test.csv --model csv_retinanet_90.pt
```
For the updated coco data-trained model, we use:
```
python visualize.py --dataset coco  --set_name_test annotation_test --coco_path /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/data_coco/ --model /pasteur/u/mzhang/surgery-hands/pytorch-retinanet/models/coco_retinanet-surg_coco_tv-final-lr=1e-05-bs=2-npt=False.pt
```

On the tracking test images, use:
```
python visualize.py --dataset csv --csv_val annotation_test_tracking.csv --csv_classes ../hand-detection-dataset/class_list.csv --model_path ./models/coco_retinanet-ego_oxford_surg-e=9-lr=1e-05-bs=4-npt=False.pt
```

## Quick Commands  
Some sample commands to quickly run:
```
# Oxford dataset training and surgery test evaluation:
python train.py --dataset csv --csv_train oxford-annotation_train_val.csv --csv_val annotation_test.csv --csv_classes ../hand-detection-dataset/class_list.csv --learning_rate 1e-05 --batch_size 2 --depth 50 --model_name oxford_train_val

# Old surgery dataset training and surgery test evaluation:
python train.py --dataset csv --csv_train old_surgery-annotation_train_val.csv --csv_val annotation_test.csv --csv_classes ../hand-detection-dataset/class_list.csv --learning_rate 1e-05 --batch_size 2 --depth 50 --model_name old_surgery_train_val
```
