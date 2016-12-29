import argparse
import os,glob
import cv2
import eval_segm as es

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--list_path', type=str, required=True)
args = parser.parse_args()

gt_dir = args.gt_dir
result_dir = args.result_dir
list_path = args.list_path

f_list = open(list_path,"r")
mIoU = 0
pixel_acc = 0
mean_acc = 0
lines = f_list.readlines()
for line in lines:
    file_name = line.rsplit("/",1)[1]
    file_name = file_name.strip("\n")
    gt_img_path = os.path.join(gt_dir,file_name)
    gt_img = cv2.imread(gt_img_path,0)

    result_img_path = os.path.join(result_dir,file_name)
    result_img = cv2.imread(result_img_path,0)
    mIoU += es.mean_IU(result_img,gt_img)
    pixel_acc +=  es.pixel_accuracy(result_img,gt_img)
    mean_acc += es.mean_accuracy(result_img,gt_img)

length = len(lines)

mIoU = mIoU/length
pixel_acc = pixel_acc/length
mean_acc = mean_acc/length

print("mIoU: %f"%(mIoU))
print("pixel_acc: %f"%(pixel_acc))
print("mean_acc: %f"%(mean_acc))