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
DEFAULT_CSV = DATA_DIR + "/raw_data.csv"

SAIL_IMAGES_PATH = "/pasteur/u/kkpatel/data/images/"
SAIL_JSON_PATH = "/pasteur/u/kkpatel/data/complete_data.json"

LOCAL_IMAGES_PATH = "/Users/stephenren/code/curis2020/MARVLous_surgery_annotator/src/images/"
LOCAL_JSON_PATH = DATA_DIR + "/raw_data.json"

# <<<<<<< Updated upstream
# AWS_DATA_PATH = "/home/ubuntu/stephen/data/marvl-surgery-annotator-validate-export.json"
# AWS_IMAGES_PATH = "/home/ubuntu/stephen/data/images/"
# =======
AWS_DATA_PATH = "/home/ubuntu/tools_data/data/marvl-surgery-annotator-validate-export.json" #new_data.json" 
AWS_IMAGES_PATH = "/home/ubuntu/tools_data/data/images/" 

# AWS_DATA_PATH = "/home/ubuntu/stephen/data/new_data.json"
# AWS_IMAGES_PATH = "/home/ubuntu/stephen/data/images/"
# >>>>>>> Stashed changes

# blacklist = ['hqeq7pZOTgY', '8LGGfKHtiPc', 'WJ2jS88EUmo', 'OXlUtv2DIzY', 'gL_7N9tvgT4', 
#     'x98rpVdG96Y', 'Mp2m1AMx8Ks', 'IJuourlriwc', 'RWHBTwfa5C8', 'thgXniYmwII', 'HW2LXjSoAc8', 
#     'mYzL383plFw', 'BWGNKectNVA', 'EgVI5o-fS6o', 'VvkCaxXqFfY', 'sHZWvYrerEs', '8VPsnDDEN04', 
#     'wDu9VyqfNu8', 'pN5bKT7U_OQ', '8eFljZ449SY', 'JUTyS7ZRRkQ', 'L8k75Onag_o', 'VsKw5d-4rq8', 
#     'pWIti7kfTyk', 'SRQb7RGQ52M', 'JjzptNEQJ2g', '-3gFrKiC99I', 'VyN4c_wsZuY']


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

    # accept_images = json.load(open('/home/ubuntu/stephen/data/acceptable_images.json'))
    # bad_images = accept_images['bad_images']
    # bad_images += bad_scalpels

    overall_data = json.load(jf)
    for key, section in overall_data.items(): # key = video #, section = annots for that video 
        #print(key) 
        if key == 'index':
            continue

        json_data = section['data'] 

        for data in json_data:
            if data["object_type"] == "image" and data["id"] is not None:  # Image annotation {, , , ... , } 
                if ignore_annotator is not None and data["original_annotator_name"] == ignore_annotator:
                    continue
                
                objects_in_image = 0
                filename = data["name"]  # Image filename 

                # if filename in bad_images:
                #     continue

                vid_id = data['video_id'] # Video ID 
                # if vid_id in blacklist:
                #     continue
                
                if acceptable is not None and vid_id not in acceptable:
                    continue
                
                img_width, img_height = Image.open(images_path + filename).size
                
                #print(hands) # False 
                if not hands and "tool_labels" in data: # ... 
                    for tool_label in data["tool_labels"]: # For each tool bbox 
                        if tool_label["category"] == "scalpel": ## Added 
                            continue 
                        if selected_tool is None or tool_label["category"] == selected_tool:
                            objects_in_image += 1
                            line = [images_path + filename]
                            line += get_coordinates(tool_label["bounding_box_position"], img_width, img_height)
                            line.append(tool_label["category"])  # Needledriver 
                            filewriter.writerow(line)

                if hands and "hand_labels" in data: # ... 
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


# bad_scalpels = ['Cczsz7JrUGU-000003389.jpg', 'Cczsz7JrUGU-000003954.jpg', 'txCYSkZIrjE-000002222.jpg', 'DpeAsOXVruw-000004031.jpg', 'AjwzlmTvT8A-000000282.jpg', 
# 'kipTlXQpPZw-000002285.jpg', 'tP2lr5QzZxM-000005847.jpg', 'SublDZXJ7p4-000006039.jpg', 'SublDZXJ7p4-000007247.jpg', 'SublDZXJ7p4-000007247.jpg', 
# 'SublDZXJ7p4-000008455.jpg', 'SublDZXJ7p4-000009663.jpg', 'TZ9IgsIMRW4-000001219.jpg', 'TZ9IgsIMRW4-000003659.jpg', 'TcYgRmsw_jg-000001943.jpg', 
# 'TcYgRmsw_jg-000002186.jpg', 'TcYgRmsw_jg-000002429.jpg', 'V6pL0fMsVn0-000002395.jpg', 'V6pL0fMsVn0-000002994.jpg', 'VL_UNwlWd3c-000004167.jpg', 
# 'VL_UNwlWd3c-000004688.jpg', 'Vnab8vZQcK8-000001987.jpg', 'Vnab8vZQcK8-000002839.jpg', 'YXAGt6GtFwE-000004872.jpg', 'wEivan1FAIA-000004049.jpg', 
# 'wiT6R0xxh7w-000002471.jpg', 'wiT6R0xxh7w-000003089.jpg', 'zJkQP-BFl8c-000002204.jpg', 'l7Fd7vTBkjE-000010478.jpg', 'lMmYj17LdbM-000001423.jpg', 
# 'lBmRC9LhFgw-000005129.jpg', 'nwqyKvQ_mNk-000002210.jpg', '1dki4N24iP8-000006245.jpg', '3Ql0fGVrQeA-000003041.jpg', '6ywlQxRzZGw-000001529.jpg', 
# '8eFljZ449SY-000001623.jpg', '8eFljZ449SY-000004871.jpg', 'BWGNKectNVA-000001772.jpg', 'BWGNKectNVA-000005909.jpg', 'GQQzVPySxPI-000001349.jpg', 
# 'ILZA2bFVGPE-000002483.jpg', 'M2PTmXVatbQ-000004189.jpg', 'OXlUtv2DIzY-000007163.jpg', 'OrK0k3sBOHY-000003365.jpg', 'OrK0k3sBOHY-000003739.jpg', 
# 'PC2SAKnlLok-000001991.jpg', 'RWHBTwfa5C8-000002979.jpg']

if __name__ == "__main__":
    main()
