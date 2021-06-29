# Surgery Hand Keypoint Analysis

Code for implementing hand keypoint analysis on our chosen model for keypoint detection on the Youtube Surgery Hands Dataset. The final model selected from our experiments (`keypoints_final.pth`) predicts 21 keypoints corresponding to the joints in a hand and the wrist. Our final model has a PCK@0.5 of 0.493 and 0.40 on the Youtube Surgery Hands dataset validation and test splits, respectively. This performance can increase if we perform filtration described below.

#### Contact
For questions about this repo, contact Krishna at kkpatel at cs dot stanford dot edu. 

## Setup

### GPU Specifications

Aquire a GPU instance with CUDA 10.1 by either requesting GPUs through [Stanford Clusters](https://github.com/yeung-lab/marvl-clusters-guide) or launching an AMI with CUDA 10.1.

### General Setup

First, follow the [instructions](https://github.com/yeung-lab/surgery-hand-detection/tree/master/deep-high-resolution-net.pytorch#setup) to setup the `deep-high-resolution-net.pytorch` project. 

Next, follow these [instructions](https://github.com/yeung-lab/surgery-hand-detection/tree/master/mask-rcnn#download-models) to download the final bounding box detection model into the `mask-rcnn/models` directory.

### Conda Environment
Follow the instructions [here](https://github.com/yeung-lab/surgery-hand-detection#install-dependencies) to set up our conda environment.

```
source activate surgery_hands

conda install -c plotly plotly=4.10.0
conda install -c plotly python-kaleido
```

### Youtube Surgery Hands Dataset
You can download the Youtube Surgery Hands Dataset from [Google Drive](https://drive.google.com/file/d/1AwOHVGvoEe3iT9nEAJP24p6sVReATccM/view?usp=sharing) and unzip it in the `deep-high-resolution-net.pytorch/data/coco/images` directory, or use the following to download the data from SAIL.

```
scp [CSID]@sc.stanford.edu:/pasteur/data/YoutubeSurgeryHands/images/all_images.zip ./
unzip all_images.zip
```
## Analyses Split By
Some of the more specific scripts below split their analyses by the following sets of categories.

### Categories 
* video quality: {good, okay, bad}
* video motion: {moving, static}
* video style: {overhead, not overhead}
* video containing severe jumps: {yes, no}
* image zoom: {hand level, close up, body level}

### Label Percentages 

* percent keypoints labeled (out of 21)
* percent keypoints occluded (out of number of keypoints labeled in that particular hand)
* percent keypoints visible (out of 21) -- commented out in scripts, as this is almost the same as percent keypoints labeled


## PCK Evaluation
For PCK evaluation, we normalize calculated distances by the maximum length of the hand bounding box's sides. We can run our PCK evaluation by using the `PCKh.py` script. 

### General
To use ground truth bounding boxes when evaluating our model, use the `--use_gt` flag, and to use the test set (versus the validation set), use the `--test` flag. We can also supply the alpha value for PCK evaluation using the `--alpha` flag (0.5 is the default). For example, to evaluate our final keypoint model's PCK@0.5 using ground truth bounding boxes on the test set, we use the following:

```
python3 PCKh.py --use_gt --test --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

If we don't use the `--use_gt` flag, the current script will load in our final bounding box model (located at `../mask-rcnn/models/bbox_final.pth`) and use predicted bounding boxes supplied by the model in our PCK evaluation.

### By Category
To split PCK evaluation split by the data quality categories mentioned above, we use the following script:

```
python3 PCK_by_cat.py --use_gt --test --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

The results are saved to a text file.

The `--test` flag and the `--use_gt` flag have the same use as above (without `--test`, we perform evaluation on the validation set, and without `--use_gt` we use our final bounding box model to detect hand bounding boxes which are then used to predict keypoints and perform PCK analysis). The `--alpha` value also remains configurable.

### By Label Percentages
When evaluating PCK by label percentages, instead of calculating a cumulative PCK, we calculate a per hand and per finger PCK, since each label percentage corresponds to an individual hand or finger. 

To explore plots like percent labeled keypoints vs hand PCK performance or percent keypoints occluded vs pointer finger PCK performance, use the following command:

```
python3 PCK_analysis.py --use_gt --test --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

This will produce interactive html plots, which are a bit prettier than what matplotlib can produce, but if you would like static versions of these graphs, there is commented out code in the script that uses matplotlib to save these graphs as images.

As with above, the `--use_gt`, `--test`, and `--alpha` flags remain configurable.

## Keypoint Accuracy
A keypoint is marked accurate or not depending on whether it is a logical distance away from its neighboring joints in proportion to the bounding box's size.

#### Threshold Generation
The thresholds accompanying such filtering can be generated by running `thresholding.py`. If you supply a configuration to `thresholding.py`, you can see whether predicted keypoints fall within a plausible distance from one another.

```
python3 thresholding.py --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

To generate these thresholds, we first found the quartiles of hand bounding box areas over all of our dataset's annotations. We then calculate the distance between every keypoint pair in the skeleton for our ground truth annotations, if both keypoints are visible. Next, we discard distances that are outliers, and compute the average distance across each quartile of hand bounding box areas. Then, to calculate the maximum and minimum deviation from this mean to be marked as average, per keypoint pair in the skeleton, we subtract the maximum distance found between these joints and the average distance to find the max threshold, and we subtract the minimum distance found between these joints and the average distance to find the minimum threshold. 

Also, if you're curious, there's a commented out section of `thresholding.py` which measures the accuracies of the ground truth annotations using the thresholds discovered, as a proof of concept.

### General
To use these thresholds to measure predicted keypoint accuracy (plausability) and filter inaccurate keypoints, use the following script:

```
python3 threshold_accuracy.py --dir ../deep-high-resolution-net.pytorch/data/coco/images/val --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml 
```
The `--dir` flag takes in the path to a directory of images that we will run our bounding box detection and hand keypoint detection models on. If we run the script with the `--strict` flag, we use strict filtration, where a keypoint is only marked as accurate if it's within a plausible distance to all of its neighbors. If the `--strict` flag is missing, we use soft filtering, which means that a keypoint is marked plausibl if it's within a plausible distance to at least one of its neighbors.

### By Category
This evaluation is the same as the corresponding PCK evaluation explained above, however, we filter out keypoints that are marked as innacurate by our calculated thresholds. 

```
python3 thresholding_pck_category_analysis.py --use_gt --test --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

As with all threshold scripts, we can use the `--strict` flag to set our filter to strict, while the default is soft. The results will be outputted into a text file.

### By Label Percentages
This evaluation is the same as the corresponding PCK evaluation explained above, however we enable keypoint filtering for keypoints that are marked as inaccurate by our calculated thresholds.

```
python3 thresholding_pck_label_analysis.py --use_gt --test --cfg ../deep-high-resolution-net.pytorch/hand_exp/keypoints_final.yaml
```

As with all threshold scripts, the default filtration is soft, but we can use the `--strict` flag to set our filter to strict. The resulting interactive visualizations will be saved to an html file, however code to save them as a static matplotlib image is commented out inside the script, in case if that is desired.
