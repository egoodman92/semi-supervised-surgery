"""
Script to sort images to the correct split folder based on annotation.csv

Ex: python sort_splits.py -f annotation_train.csv
"""

import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
args = parser.parse_args()

img_src = 'img'

with open(args.filename, 'r') as f:
    data = csv.reader(f)
    for row in list(data):
        try:
            split, img_name = row[0].split('/')  # e.g. train/breast-...
            # Move files
            try:
                os.rename('{}/{}'.format(img_src, img_name), '{}/{}'.format(split, img_name))
            except FileNotFoundError:
                print('Could not find file {}'.format(img_name))
        except Exception as e:
            if 'name' in row[0]:
                pass
            else:
                raise e
