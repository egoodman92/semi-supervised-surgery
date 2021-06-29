import json
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SAIL_JSON_PATH = "/pasteur/u/rensteph/curis2020/data/new_data.json"

sf = open(PROJECT_ROOT + '/statistics.txt', 'w')

def publish(s):
    print(str(s))
    print(str(s), file=sf)

df = open(SAIL_JSON_PATH)

json_data = json.load(df)['data']

image_count = {'total': 0, 'with_tools': 0, 'bovie': 0, 'scalpel': 0, 'needledriver': 0, 'forceps': 0}
tool_count = {'bovie': 0, 'scalpel': 0, 'needledriver': 0, 'forceps': 0}

for data in json_data:
    if data['object_type'] == 'image':
        image_count['total'] += 1
        saw_tools = {}
        if 'tool_labels' in data:
            num_tools = 0
            for tool_label in data['tool_labels']:
                num_tools += 1
                category = tool_label['category']
                tool_count[category] += 1
                saw_tools[category] = 1

            if num_tools > 0:
                image_count['with_tools'] += 1

        for tool_seen in saw_tools:
            image_count[tool_seen] += 1

publish('Total number of images: {}'.format(str(image_count['total'])))
publish('Images with tools: {}'.format(str(image_count['with_tools'])))
publish('Without tools: {}'.format(str(image_count['total'] - image_count['with_tools'])))
publish('Images with bovie: {}'.format(str(image_count['bovie'])))
publish('Images with scalpel: {}'.format(str(image_count['scalpel'])))
publish('Images with needledriver: {}'.format(str(image_count['needledriver'])))
publish('Images with forceps: {}'.format(str(image_count['forceps'])))

publish('\nTool instances:')
publish('Bovie occurances: {}'.format(str(tool_count['bovie'])))
publish('Scalpel occurances: {}'.format(str(tool_count['scalpel'])))
publish('Needledriver occurances: {}'.format(str(tool_count['needledriver'])))
publish('Forceps occurances: {}'.format(str(tool_count['forceps'])))


        
