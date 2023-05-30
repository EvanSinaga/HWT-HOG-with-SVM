import os
import csv
# import pandas as pd
from natsort import natsorted
# from Helper import getCsvPath
from SaveROI import segmented_path


iter_path = os.scandir(segmented_path)
gt_path = os.path.join(segmented_path, "GroundTruth.csv")

dir_path = []
dir_name = []
for i, entry in enumerate(iter_path):
    if entry.is_dir():
        dir_path.append(entry.path)
        dir_name.append(entry.name)

img_list_0 = natsorted(os.listdir(dir_path[0]))  # Sorting negative images
img_list_1 = natsorted(os.listdir(dir_path[1]))  # Sorting positive images

# Annotate non-augmented images
print("Start annotating non-augmented images")
with open(gt_path, 'r') as f, \
        open(os.path.join(dir_path[0], "GT-{}.csv".format(dir_name[0])), 'w') as f0, \
        open(os.path.join(dir_path[1], "GT-{}.csv".format(dir_name[1])), 'w') as f1:
    gt_reader = csv.DictReader(f, delimiter=',')
    gt_header = gt_reader.fieldnames

    # gt_reader.__next__()  # Skip header
    writer0 = csv.DictWriter(f0, fieldnames=gt_header, lineterminator='\n')
    writer1 = csv.DictWriter(f1, fieldnames=gt_header, lineterminator='\n')
    writer0.writeheader()
    writer1.writeheader()

    for gt_row in gt_reader:
        filename = str(gt_row["Filename"])
        if filename in img_list_0:
            gt_row.update({"ClassId": "0"})
            writer0.writerow(gt_row)
        elif filename in img_list_1:
            gt_row.update({"ClassId": "1"})
            writer1.writerow(gt_row)
    print("GT-{}.csv created.".format(dir_name[0]))
    print("GT-{}.csv created.".format(dir_name[1]))
