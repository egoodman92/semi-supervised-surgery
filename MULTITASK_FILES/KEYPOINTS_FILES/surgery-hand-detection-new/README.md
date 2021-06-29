# Surgery Hand and Keypoint Detection Models

Code for implementing hand bounding box detection and hand keypoint detection on the Youtube Surgery dataset. 


## Bounding Box Detection

Bounding box detection is done using a Faster-RCNN trained using Facebook's [detectron2](https://github.com/facebookresearch/detectron2) package. Our final bounding box detection model scores a mAP of 80.942 and 82.388 on the Youtube Surgery Hands dataset validation and test splits, respectively. The final model, ``bbox_final.pth``, and can be downloaded [here](https://drive.google.com/file/d/125iWaI4Mn_-kjjExQVQZJXIKviryaCqw/view?usp=sharing). More details on how to run this model separately can be found [here](https://github.com/yeung-lab/surgery-hand-detection/tree/master/mask-rcnn#surgery-hand-and-keypoint-detection-models).

## Keypoint Detection

Keypoint detection is done using the Deep High Resolution Networks framework trained using [Microsoft's framework](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) package. Our final model has a PCK@0.5 of 0.493 and 0.40 on the Youtube Surgery Hands dataset validation and test splits, respectively. However, this performance is higher once we filter out keypoint predictions using rudimentary hand skeleton analysis in `keypoint-analysis`. The final model, ``keypoints_final.pth``, and can be downloaded [here](https://drive.google.com/file/d/1KNizN8bsBFFyiAX2CT9bIZYKlsUN3UgO/view?usp=sharing). More details on how to run this model separately can be found [here](https://github.com/yeung-lab/surgery-hand-detection/tree/master/deep-high-resolution-net.pytorch#surgery-hand-keypoint-detection-model).

### Performance and Analysis

Keypoint performance (PCK) and analyses can be evaluated by running scripts found in the `keypoint-analysis` directory. Predictably, hand keypoint detection improves when the camera is stable (from an average PCK@0.5 of 37% to 41%), when the camera is mounted overhead and showing whole hands (from an average PCK@0.5 of 38.2% to 54.7%), and when there are fewer severe jumps in the video (from an average PCK@0.5 of 37.6% to 49.8%). 

## Joint Tracking and Analysis

See `hand-tracking` folder for more details.


### Future Work
Future work for Hand tracking and analysis includes:
* Fine tuning SORT parameters using multiple validation videos
* Custom training DeepSORT for surgery hand bounding box tracking
* Keypoint smoothing: if keypoint detections move drastically between frames, throw them out or average them
* Hand pose classification


## Annotations
See `annotations` for all labels used after each round of annotation for the Youtube Surgery dataset. Also, see [here](https://github.com/yeung-lab/surgery-hand-detection/tree/master/annotations#annotator-agreement) for summary annotator agreement statistics on hand bounding box and hand keypoint annotations.

## Directory Organization

* `mask-rcnn`: detectron2 experiments for surgery hand bounding box and keypoint detection (final bounding box detection model is this architecture)
* `deep-high-resolution-nets.pytorch`: DHRN experiments for surgery hand keypoint detection (final keypoint detection model is this architecture)
* `hand-tracking`: Hand tracking script and accompanying analyses that are analygous to selected OSATS criteria
* `keypoint-analysis`: Accuracy and PCK analysis scripts to evaluate the final keypoint detection model using either ground truth bounding box annotations or bounding box predictions from our final model
* `annotations`: All original versions of the annotations on the Surgery Hands dataset, as well as annotator agreement statistics
* `PoseWarper`: Preliminary experimentation with the PoseWarper framework indicates that it's not a good fit for our research (yet), this may change if we can collect a large amount of additional high quality keypoint data, framework primarily functions to make marginal improvements on a high accuracy base model
* `retinanet`: old experiments with RetinaNet 


## Visualizing Performance
NOTE: the following script visualizes the frame-by-frame detections of both the hand bounding box and hand keypoint detection models only. To see a visualization that filters out detections unassociated with tracked hands, see work in the  `hand-tracking` directory.

#### Install Dependencies
Ensure that you are not in a different conda environment first.
```
conda env create --file environment.yml
conda activate surgery_hands
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

#### Extra Setup

If you have not yet followed the [README to set up the `deep-high-resolution-net.pytorch`](https://github.com/yeung-lab/surgery-hand-detection/tree/master/deep-high-resolution-net.pytorch#surgery-hand-keypoint-detection-model) project, do the following, otherwise, skip to the next section.

First, make the libraries:
```
cd deep-high-resolution-net.pytorch/lib
make
```

Next, install [COCOAPI](https://github.com/cocodataset/cocoapi) in the `deep-high-resolution-net.pytorch` directory.
```
cd deep-high-resolution-net.pytorch

git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI
python3 setup.py install --user
```


#### Final Model Weights
Download model weights (`bbox_final.pth` and `keypoints_final.pth`) from [Google Drive](https://drive.google.com/drive/folders/1CIeX9HCtua9RGTQZJa_Zl7oJ6FzgDd4U?usp=sharing) and save weights to this directory or:

```
wget https://stanford-marvl.s3.amazonaws.com/hands/bbox_final.pth
wget https://stanford-marvl.s3.amazonaws.com/hands/keypoints_final.pth
```

### Video
Download a video to analyze (following video is not from our dataset)

```
pip install youtube-dl
youtube-dl https://www.youtube.com/watch?v=dIdBaYA4s4o -f 22 -o video.mp4
```

Or, download a video from our dataset: 

```
wget https://marvl-surgery.s3.amazonaws.com/videos/0EeZIRDKYO4.mp4
```

Then, run inference on the video using the following command.

```
python3 demo.py --bb_cfg bbox.yaml --video video.mp4 --produce_vid --cfg keypoints.yaml 
```

To only generate the predictions JSON file, do not provide the `--produce_vid` flag. The keypoint detection model configuration must be passed in last. If needed, you can modify the configuation in the command line as well:

```
python3 demo.py --bb_cfg bbox.yaml --video video.mp4 --produce_vid --cfg keypoints.yaml TEST.MODEL_FILE <path to model>
```

#### Notes

Resulting JSON file is in the format Will needs to overlay detections on action recognition visualization. To create a JSON of predictions and bounding box scores (required for our hand tracking script), add the `--tracking` flag.

## Resources

Contact Krishna at kkpatel at cs dot stanford dot edu

For a guide on the COCO Data Format, see [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).

For a guide on mAP COCO Evaluation, see [here](https://cocodataset.org/#detection-eval).

For the definition of PCK, see [here](https://arxiv.org/abs/1704.07809). Note, we use PCK normalized by the size of the hand bounding box (analogous to PCKh, PCK normalized by the size of a head bounding box, which is used to evaluate human keypoint detection).
