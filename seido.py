import cv2 
import numpy as np
import os
from PIL import Image
import math
from numpy.lib.function_base import blackman
import glob

# result_dir = '/home/takahashi/takahashi/works/practice/pytorch-nested-unet/outputs/real_datasets_UNet_woDS/0/'
# lbl_dir = './check_PET/1/'
#out_dir='./result/unet_result/'
dir = "/home/takahashi/takahashi/works/practice/pytorch-nested-unet/outputs/de_dataset1"

# lbl_list = os.listdir(lbl_dir)
result = glob.glob(dir + "**/*.jpgpre.bmp")
lbl    = glob.glob(dir + "**/*.jpgmask.bmp")
result.sort()
lbl.sort()

'''
black = 0
yellow = 0
orange = 0
magenta = 0 
blue  = 0
rad = 0
cyan = 0
green = 0
purple = 0
'''


AV_IoU = 0.0
b_IoU = []
b_Dice = []
b_Acc = []

for i in range(len(lbl)):
    # lbl_name = lbl_list[i]
    print(i+1)
    # print(lbl_name)
    print(lbl[i])
    print(result[i])
    label = cv2.imread(lbl[i], cv2.IMREAD_GRAYSCALE)
    res = cv2.imread(result[i], cv2.COLOR_BGR2GRAY) 
    #black_map = np.zeros((512,512,3))
    print(label.shape)
    print(res.shape)



    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp = 0.0
    IoU = 0.0
    Acc = 0.0
    Dice = 0.0
    Precision = 0.0
    Recall = 0.0

    for x in range(label.shape[0]):
        for y in range(label.shape[1]):

            if label.item(y,x)== 0 and res.item(y,x) == 0:
                tn += 1
            elif label.item(y,x) == 0 and res.item(y,x) == 255:
                fp += 1
            elif label.item(y,x) == 255 and res.item(y,x) == 0:
                fn += 1
            elif label.item(y,x) == 255 and res.item(y,x) == 255:
                tp +=1
    print(tp, tn, fp, fn)
    Acc = (tp+tn)/(tp+fp+fn+tn)
    IoU =  tp/(tp+fp+fn)
    Dice = tp/(tp + (0.5*fp + 0.5*fn))
    b_Acc.append(Acc)
    b_IoU.append(IoU)
    b_Dice.append(Dice)
            

    print('IoU:%5f  Dice:%5f  Acc:%5f' % (IoU, Dice, Acc))

print('result')
print('IoU:%f  Dice:%f  Acc:%f' % (sum(b_IoU) / len(b_IoU), sum(b_Dice) / len(b_Dice), sum(b_Acc) / len(b_Acc)))

