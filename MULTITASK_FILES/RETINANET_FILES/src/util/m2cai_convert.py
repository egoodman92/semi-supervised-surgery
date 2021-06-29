import csv
import os
import xmltodict
from pathlib import Path

DATA_DIR = str(Path(__file__).resolve().parents[1]) + "/data"
SAIL_DATA_PATH = '/pasteur/u/rensteph/curis2020/m2cai16-tool-locations/'
SAIL_IMAGES_PATH = SAIL_DATA_PATH + 'JPEGImages/'
SAIL_ANNOTATIONS_PATH = SAIL_DATA_PATH + 'Annotations/'
DEFAULT_CSV = DATA_DIR + "/raw_data.csv"


def make_annotation(annotation, img_name, writer):
    box = annotation['bndbox']
    writer.writerow([SAIL_IMAGES_PATH + img_name, box['xmin'], box['ymin'],
                     box['xmax'], box['ymax'], annotation['name']])


def main():
    out_f = open(DEFAULT_CSV, 'w')
    csv_writer = csv.writer(out_f, delimiter=',')

    for _, _, files in os.walk(SAIL_ANNOTATIONS_PATH):
        for file in files:
            data = xmltodict.parse(open(SAIL_ANNOTATIONS_PATH + file).read())['annotation']
            img_name = data['filename']
            if 'object' in data:
                data = data['object']
                if isinstance(data, dict):
                    make_annotation(data, img_name, csv_writer)

                elif isinstance(data, list):
                    for obj in data:
                        make_annotation(obj, img_name, csv_writer)


if __name__ == "__main__":
    main()
