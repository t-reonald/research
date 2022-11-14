from pathlib import Path
import cv2
import numpy as np

def search_result(src_path, mask_path, pre_path):
    #ct画像のpath
    src_paths = [str(p) for p in Path(src_path).glob("**/*.bmp")]
    #mask画像のpath
    mask_paths = [str(p) for p in Path(mask_path).glob("**/*.bmp")]
    #predict画像のpath
    pre_paths = [str(p) for p in Path(pre_path).glob("**/*.jpg")]
    print(len(pre_paths))

    pre_mask = []

    for i in pre_paths:
        i_n = i.split("/")[-1]
        i_x = i_n.replace('.jpg', '')
        for j in mask_paths:
            j_n = j.split("/")[-1]
            j_x = j_n.replace('.bmp', '')
            if i_x == j_x:
                for k in src_paths:
                    k_n = k.split("/")[-1]
                    k_x = k_n.replace('.bmp', '')
                    if j_x == k_x:
                        key = ["pre", "mask", "src"]
                        value = [i, j, k]
                        pre_mask.append(dict(zip(key, value)))

    return pre_mask


#検出できたところを赤、未検出を青、過検出を緑で表示する関数、そしてその表示した画像のmaskと予想、そして元のCT画像を一緒に保存する関数。    
def make_mask(dicsionaries, save_path):

    for dict in dicsionaries:
        pre_path = dict["pre"]
        mask_path = dict["mask"]
        src_path = dict["src"]
        i = str(pre_path).split("/")[-1]
        save = i.replace('.bmp', '')
        pre_img = cv2.imread(pre_path)
        gray_pimg = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        _, binpre_img = cv2.threshold(gray_pimg, 128, 255, cv2.THRESH_BINARY)
        mask_img = cv2.imread(mask_path)
        gray_mimg = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        src_img = cv2.imread(src_path)
        com_img = np.zeros((512, 512, 3), np.uint8)
        for x in range(512):
            for y in range(512):
                if binpre_img.item(y,x) == 0 and gray_mimg.item(y,x) == 0:
                    com_img[y,x,0] = 0
                    com_img[y,x,1] = 0
                    com_img[y,x,2] = 0
                elif binpre_img.item(y,x) == 255 and gray_mimg.item(y,x) == 255:
                    com_img[y,x,0] = 0
                    com_img[y,x,1] = 0
                    com_img[y,x,2] = 255
                elif binpre_img.item(y,x) == 255 and gray_mimg.item(y,x) == 0:
                    com_img[y,x,0] = 0
                    com_img[y,x,1] = 255
                    com_img[y,x,2] = 0
                elif binpre_img.item(y,x) == 0 and gray_mimg.item(y,x) == 255:
                    com_img[y,x,0] = 255
                    com_img[y,x,1] = 0
                    com_img[y,x,2] = 0
        cv2.imwrite(save_path + str(save)+ "_color.bmp", com_img)
        cv2.imwrite(save_path + str(save)+ "src.bmp", src_img)
        cv2.imwrite(save_path + str(save) + "pre.bmp", binpre_img)
        cv2.imwrite(save_path + str(save) + "mask.bmp", mask_img)


def main():
    src_path = '/home/takahashi/takahashi/works/practice/pytorch-nested-unet/inputs/validation_real/dataset1/val/images'
    mask_path = '/home/takahashi/takahashi/works/practice/pytorch-nested-unet/inputs/validation_real/dataset1/val/masks/0'
    pre_path = '/home/takahashi/takahashi/works/practice/pytorch-nested-unet/outputs/validation_real_UNet_up_deform_woDS/0'
    save_path = '/home/takahashi/takahashi/works/practice/pytorch-nested-unet/outputs/de_dataset1/'
    dict = search_result(src_path, mask_path, pre_path)
    make_mask(dict, save_path)


if __name__ == '__main__':
    main()
