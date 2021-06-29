# Youtube Surgery Dataset Raw Annotations
Raw annotations for the Youtube Surgery Dataset.

## Directory Organization

* `original annotations`: detectron2 experiments for surgery hand bounding box and keypoint detection (final bounding box detection model is this architecture)
* `deep-high-resolution-nets.pytorch`: DHRN experiments for surgery hand keypoint detection (final keypoint detection model is this architecture)
* `tools second pass`: Hand tracking script and accompanying analyses that are analygous to selected OSATS criteria
* `validated`: Accuracy and PCK analysis scripts to evaluate the final keypoint detection model using either ground truth bounding box annotations or bounding box predictions from our final model

## Folder Contents

### Original Annotations
* `marvl-surgery-annotation-export.json`: annotations completed by our Boston collaborators through the MARVL Surgery Annotator website (roughly half of the images)
* `rohan_4.8._v2.json`: Rohan's pass through the Stanford team's annotations.
* `stephen.json`: Stephen's pass through the Stanford team's annotations.
* `rohan_150_part1.json`: Part 1 of missing annotations from Stanford half. 
* `rohan_150_part2.json`: Part 2 of missing annotations from Stanford half.

Combined, we have a fully annotated dataset. Some of the annotations in `stephen.json` and `rohan_4.8._v2.json` overlap, and 30 videos in our half weren't annotated, hence `rohan_150_part1.json` and `rohan_150_part2.json`.

### Tools Second Pass
Second pass over tools annotations only to catch any unlabled / mislabeled tools. 

* `rohan_data.json`: images with image id >= 1716 were re-examined in this file
* `stephen_data.json`: images with image id < 1716 were re-examined in this file

### Validated

* `marvl-surgery-annotator-validate-export.json`: validated dataset produced through the MARVL Surgery Annotator website


## Annotator Agreement
Calulated a set of statistics describing annotator agreement for hand bounding box and hand keypoint annotations. Run the script using the following command (be sure to have all Youtube Surgery images located at `../all_images`):

```
python3 annotator_agreement.py
```

Results from running the script are featured below:

### Bounding Boxes

* 211 new hand bboxes
* 20 hand bboxes deleted
* average IOU between original and validated bboxes 0.997
* 76 bboxes changed
* 5,918 bboxes unchanged (dimensions were untouched)

#### Stats
* 98.4% of original bounding boxes were untouched (dimensions unchanged, not deleted)
* average IOU between original bboxes that were changed in validation and validation bboxes 0.80


### Keypoints

|                                                                  | 1   | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11  | 12    | 13    | 14    | 15  | 16  | 17    | 18  | 19  | 20  | 21  |
|------------------------------------------------------------------|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----|-------|-------|-------|-----|-----|-------|-----|-----|-----|-----|
| average distance from original                                   | 0.0 | 0.227 | 0.097 | 0.344 | 0.056 | 0.074 | 0.127 | 0.108 | 0.062 | 0.164 | 0.0 | 0.087 | 0.143 | 0.009 | 0.0 | 0.0 | 0.020 | 0.0 | 0.0 | 0.0 | 0.0 |
| average distance from original (only counting changed keypoints) |     | 44.10 | 18.15 | 73.02 | 20.85 | 12.27 | 25.16 | 28.57 | 18.44 | 29.26 |     | 52.32 | 47.39 | 4.04  |     |     | 8.93  |     |     |     |     |
| keypoints changed (not including deleted keypoints)              | 0   | 2     | 3     | 3     | 2     | 4     | 4     | 3     | 3     | 3     | 0   | 1     | 2     | 1     | 0   | 0   | 1     | 0   | 0   | 0   | 0   |
| new keypoints                                                    | 1   | 9     | 15    | 15    | 5     | 19    | 13    | 17    | 7     | 15    | 6   | 9     | 6     | 12    | 8   | 8   | 3     | 5   | 6   | 2   | 0   |
| deleted keypoints                                                | 3   | 2     | 1     | 2     | 2     | 0     | 0     | 1     | 1     | 1     | 3   | 1     | 2     | 3     | 2   | 2   | 1     | 3   | 2   | 2   | 2   |

### Stats
* 99.39% keypoints unchanged 
* 32 keypoints changed
* 36 keypoints deleted
* 181 keypoints added
* 11,085 keypoints unchanged



