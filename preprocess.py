import os
import cv2
import json
import numpy as np

os.makedirs('data/Platelets', exist_ok=True)
os.makedirs('data/WBC', exist_ok=True)
os.makedirs('data/RBC', exist_ok=True)

os.makedirs('none', exist_ok=True)

ann_dir = os.path.join('bddc', 'ann')
img_dir = os.path.join('bddc', 'img')

for fname in os.listdir(ann_dir):
	old_file_path = os.path.join(ann_dir, fname)
	new_file_path = os.path.join(ann_dir, fname.split('.')[0] + '.json')

	os.rename(old_file_path, new_file_path)

for fname in os.listdir(ann_dir):
	file_path = os.path.join(ann_dir, fname)
	with open(file_path, 'r') as f:
		data = json.load(f)

	img_path = os.path.join(img_dir, fname[:-5] + '.jpeg')
	image = cv2.imread(img_path)

	for obj in data['objects']:
		idf = obj['id']
		class_title = obj['classTitle']
		[x_min, y_min], [x_max, y_max] = obj['points']['exterior']

		if x_min >= x_max or y_min >= y_max:
			continue

		img = image[y_min:y_max, x_min:x_max]

		save_dir = os.path.join('data', class_title)
		save_path = os.path.join(save_dir, f'{idf}.jpeg')

		cv2.imwrite(save_path, img)