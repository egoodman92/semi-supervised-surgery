"""
Script for converting json annotations in to csv format for training
Only takes into consideration tool boundary boxes 

# Re-implementation from new git clone surgery tool detection 
# 
"""
import csv
import json
import argparse
from pathlib import Path
from PIL import Image

DATA_DIR = str(Path(__file__).resolve().parents[1]) + "/data"
DEFAULT_CSV = DATA_DIR + "/raw_data2.csv" #

SAIL_IMAGES_PATH = "/pasteur/u/kkpatel/data/images/"
SAIL_JSON_PATH = "/pasteur/u/kkpatel/data/complete_data.json"

LOCAL_IMAGES_PATH = "/Users/stephenren/code/curis2020/MARVLous_surgery_annotator/src/images/"
LOCAL_JSON_PATH = DATA_DIR + "/raw_data.json"

AWS_DATA_PATH = "/home/ubuntu/tools_data/data/marvl-surgery-annotator-validate-export.json" #new_data.json" 
AWS_IMAGES_PATH = "/home/ubuntu/tools_data/data/images/" 

# AWS_DATA_PATH = "/home/ubuntu/stephen/data/new_data.json"
# AWS_IMAGES_PATH = "/home/ubuntu/stephen/data/images/"


def get_coordinates(position, img_width, img_height):
    left = position["left"]
    top = position["top"]
    width = position["width"]
    height = position["height"]
    x1 = int(float(left) * img_width)
    y1 = int(float(top) * img_height)
    x2 = int((float(left) + float(width)) * img_width)
    y2 = int((float(top) + float(height)) * img_height)

    return [str(x1), str(y1), str(x2), str(y2)]


def convert(images_path, json_path, selected_tool, ignore_negatives, acceptable, ignore_annotator, hands, ignore_chirality):
    jf = open(json_path)
    cf = open(DEFAULT_CSV, 'w')
    filewriter = csv.writer(cf, delimiter=',')

    json_data = json.load(jf)['0'] #'data']
    for data in json_data:
        if data["object_type"] == "image" and data["id"]:
            if ignore_annotator is not None and data["original_annotator_name"] == ignore_annotator:
                continue
            
            objects_in_image = 0
            filename = data["name"]
            vid_id = data['video_id']
            
            if acceptable is not None and vid_id not in acceptable:
                continue
            
            img_width, img_height = Image.open(images_path + filename).size
            
            if not hands and "tool_labels" in data:
                for tool_label in data["tool_labels"]:
                    if tool_label["category"] == "scalpel": 
                        continue 
                    if selected_tool is None or tool_label["category"] == selected_tool:
                        objects_in_image += 1
                        line = [images_path + filename]
                        line += get_coordinates(tool_label["bounding_box_position"], img_width, img_height)
                        line.append(tool_label["category"])
                        filewriter.writerow(line)

            if hands and "hand_labels" in data:
                for hand_label in data["hand_labels"]:
                    objects_in_image += 1
                    line = [images_path + filename]
                    line += get_coordinates(hand_label["bounding_box_position"], img_width, img_height)
                    line.append('hand' if ignore_chirality else hand_label['chirality'])
                    filewriter.writerow(line)

            # Case were tools are not present in the image - add negative label
            if not ignore_negatives and objects_in_image == 0:
                filewriter.writerow([images_path + filename, '', '', '', '', ''])
                

def build_acceptable_videos(json_path):
    data_f = open(json_path)
    json_data = json.load(data_f)['data']

    acceptable = []
    for data in json_data:
        if data['object_type'] == 'video':
            if data['quality'] == 'good' or data['quality'] == 'okay':
                acceptable.append(data['id'])

    return acceptable
    

def main():
    parser = argparse.ArgumentParser(description='Script to convert data into csv format for pytorch-retinanet.')

    parser.add_argument('--datapath', help='Path to json annotations')
    parser.add_argument('--imagepath', help='Path to image directory')
    parser.add_argument('--use_local', help='Use pre-loaded LOCAL_IMAGES_PATH (check convert_data.py)', action="store_true")
    parser.add_argument('--focus_tool', help='Only use annotations for one particular tool')
    parser.add_argument('--quality_control', action='store_true')
    parser.add_argument('--ignore_negatives', action='store_true')
    parser.add_argument('--ignore_annotator')
    parser.add_argument('--hands', action='store_true')
    parser.add_argument('--ignore_chirality', action='store_true')
    parser.add_argument('--aws', action='store_true')

    args, leftover = parser.parse_known_args()

    images_path = SAIL_IMAGES_PATH
    json_path = SAIL_JSON_PATH

    if args.imagepath is not None:
        images_path = args.imagepath

    if args.datapath is not None:
        json_path = args.datapath

    if args.use_local:
        images_path = LOCAL_IMAGES_PATH
        json_path = LOCAL_JSON_PATH

    if args.aws:
        images_path = AWS_IMAGES_PATH
        json_path = AWS_DATA_PATH
        
    tool = args.focus_tool
    if tool is not None:
        print("Focusing on tool: " + tool)

    acceptable_videos = None
    if args.quality_control:
        acceptable_videos = build_acceptable_videos(json_path)
        
    print("Converting json data from " + json_path)
    convert(images_path, json_path, tool, args.ignore_negatives, acceptable_videos, args.ignore_annotator, args.hands, args.ignore_chirality)
    print("Converted data saved under " + DEFAULT_CSV)


if __name__ == "__main__":
    main()
