# Surgery Action Recognition

## Overview
This repository consists of the PyTorch library used for action recognition on the Youtube Surgery dataset.

## Setup

### Virtual environment
**Note**: This repository used PyTorch 1.4. 

```bash
# Create a new environment.
conda create -n pytorch1.4 python=3.6

# Activate the enviroment.
conda activate pytorch1.4

# Install dependencies
pip install -r requirements.txt
```


## Data
The Youtube Surgery dataset consists of ~390 open-surgery videos curated from YouTube. A subset of these videos were annotated for three surgical actions: cutting, tying, and suturing.

### Downloading videos 
```
aws s3 sync s3://marvl-surgery/videos data/videos
```




