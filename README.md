# open surgery multitask-model
This repo contains implementation details on training and performing inference with our multitaskmodel model. The full paper can be read at   https://docs.google.com/document/d/1waRRG5i78uHqpU0Tvc-I4MQaVndIu4v1fRKqGGWKBiY/edit

<p align="center">
  <img width="680" height="280" src="multitask-inference.jpg">
</p>

### Directory Setup
The following is the directory setup used on our GCP virtual machines:

```
├── data
│   ├── images (3431 .jpg images)
│   ├── videos (339 .mp4 videos)
│   ├── jsons (contains different versions of full data)
│   ├── annotations (after Rohan and Emmett re-editing)
│   ├── notebooks_for_processing_annotations (used to edit annotations if needed)
│   ├── surgical_signatures (video data from Youtube Open Surgery Dataset curated by specialty)
│   ├── skill_signatures (video data from BI used for skill analysis)

├── notebooks 
│   ├── training_multitaskmodel.ipynb (most important!)
│   ├── multitask-performance.ipynb (simple set of commands to get accuracy on train/val/test for actions and detections!)
│   ├── track-hands-and-skills.ipynb (used to move from _tracked.json to interpretable kinematics)
│   ├── example_inference_graphic.ipynb (gives action bar)
│   ├── surgical_specialties_from_YOSD.ipynb (filters from YOSD to create ytid list)
│   ├── realtime_analysis.ipynb (performs aggregate realtime analysis on a folder)
│   ├── interpretable_LDA.ipynb (used to interpret data from surgical_signatures)

├── scripts
│   ├── inference.py (perform multitask inference on a video with 'python inference.py' or 'python inference.py --keypoints')
│   ├── skill-sort.py (used for tracking with 'python skill-sort.py')

├── MULTITASK_FILES 
│   ├── RETINANET_FILES (copied from Rohan/Stephen's retinanet surgery repo)
│   ├── TSM_FILES (copied from Will's TSM repo)
│   ├── KEYPOINT_FILES (copied from Krishna's TSM repo)

├── logs
│   ├── contains folders for each experiment, with a .pt checkpoint, as well as file to read in tensorboard

├── produced_videos
│   ├── empty folder to put videos in for ease of inference

```

## Data

For training, we used [64, 3, 352, ~576] for actions, and [2, 3, 352, ~576] for detections\
For inference, we use [64, 3, 352, ~576]


## Inference and Tracking Scripts

While the 'multitaskenv' conda environment is activated, running inference and tracking is as simple as performing the following two commands. Note: the target video appendectomy.mp4 has to be in ./produced_videos/, otherwise one needs to specify another argument --dir='/a/b/c/', the location of the target appendectomy.mp4 video. Tracking will not work if keypoints are not present in the produced appendectomy_detections.mp4/.json file. Faster real-time inference can be performed by removing the '--keypoints' argument from the inference script, but tracking will not work. Other additionally helpful arguments can be found in the scripts themselves.

```bash
$ python inference.py --vid_name=appendectomy.mp4 --keypoints
$ python skill_sort.py --vid_name=appendectomy.mp4
```

## Environment

The environment used in this work is called 'multitaskenv', and is used across training, inference, and tracking. The environment is quite sensitive to many things: I found cudatoolkit=11.1, pytorch=1.8.0, torchvision=0.9.0, and detectron built  on these parameters have to be very carefully decided. This may break/be troubling for other GPUs that don't use CTK=11.1. The environment can be found in the environment.yml file via ```bash conda env create -f environment.yml```.


## Buckets

gs://surgery-multitask-images\
gs://surgery-multitask-videos-and-cache (contains raw and cached frames too!)\
gs://surgery-multitask-model-weights (contains raw and cached frames too!)\
gs://surgery-multitask-jsons
