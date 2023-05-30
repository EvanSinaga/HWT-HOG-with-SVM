import sys
import csv
import math

from Helper import *
from Segmentation import *
from timeit import default_timer as timer


segmented_dir = "Segmented"
segmented_path = os.path.join(datapath, segmented_dir)
# Check directory and create
checkDirectory(segmented_dir)


def main():
    print("Starting extraction ...")
    start = timer()
    total_files = len(imglist)
    with open(segmented_path + '/GroundTruth.csv', 'w') as f:
        fieldnames = ['Filename', 'Origin', 'w', 'h', 'xmin', 'ymin', 'xmax', 'ymax', 'ClassId']
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        # write multiple rows
        for i, entry in enumerate(imglist):
            # Read image
            readimg = cv2.imread(entry.path)
            # Preprocess image
            larged = enlarge(readimg, 200)
            # Convert color
            rgbimg = convertColor(larged, "rgb")
            hsvimg = convertColor(larged, "hsv")

            # Applied Segmentation
            morphresult = ColorSegmentation(rgb=rgbimg, hsv=hsvimg).morphing()

            # List of cropped ROI
            roi_list = ImageRoi(morphresult).coordinates(rgbimg)[0]
            # List of ROI coordinates
            gt_roi = ImageRoi(morphresult).coordinates(rgbimg)[1]

            for j, roi in enumerate(roi_list):
                title = "Image_{:05d}_{}.png".format(1 + i, 1 + j)
                if os.path.exists(title):
                    pass
                else:
                    # pass
                    # Create .png image
                    roi_rgb = convertColor(roi, "rgb")
                    cv2.imwrite(os.path.join(segmented_path, title), roi_rgb)
                    # Write to CSV
                    writer.writerow({"Filename": str(title),
                                     "Origin": str(entry.name),  # os.scandir file name
                                     "w": gt_roi[j][0],
                                     "h": gt_roi[j][1],
                                     "xmin": gt_roi[j][2],
                                     "ymin": gt_roi[j][3],
                                     "xmax": gt_roi[j][4],
                                     "ymax": gt_roi[j][5]})

            # Progress Bar
            counter = (i + 1) / total_files
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * counter), 100 * counter))
            sys.stdout.write(" | Current : {} |".format(entry.name))
            sys.stdout.flush()

        end = timer()
        print("\nProcesses Completed!")
        ttime = end - start  # total time
        hours = math.floor(ttime // 3600)
        minutes = math.floor((ttime // 60) % 60)
        seconds = ttime % 60
        print("Time elapsed = {}:{}:{}".format(hours, minutes, seconds))


if __name__ == "__main__":
    main()
