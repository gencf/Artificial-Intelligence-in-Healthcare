# This program is written by Abubakr Shafique (abubakr.shafique@gmail.com) 
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import os
import sys

def Dicom_to_Image(DCM_Img, Window_Min, Window_Max):

    rows = DCM_Img.get(0x00280010).value #Get number of rows from tag (0028, 0010)
    cols = DCM_Img.get(0x00280011).value #Get number of cols from tag (0028, 0011)

    if (DCM_Img.get(0x00281052) is None):
        Rescale_Intercept = 0
    else:
        Rescale_Intercept = int(DCM_Img.get(0x00281052).value)

    if (DCM_Img.get(0x00281053) is None):
        Rescale_Slope = 1
    else:
        Rescale_Slope = int(DCM_Img.get(0x00281053).value)

    New_Img = np.zeros((rows, cols), np.uint8)
    Pixels = DCM_Img.pixel_array

    for i in range(0, rows):
        for j in range(0, cols):
            Pix_Val = Pixels[i][j]
            Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept
            if (Rescale_Pix_Val > Window_Max): #if intensity is greater than max window
                New_Img[i][j] = 255
            elif (Rescale_Pix_Val < Window_Min): #if intensity is less than min window
                New_Img[i][j] = 0
            else:
                New_Img[i][j] = int(((Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) * 255) #Normalize the intensities
    return New_Img

def convert_hu(hu, lower_bound=0, upper_bound=255):
    hu[hu>upper_bound] = 0
    hu[hu<lower_bound] = 0
    return hu

def draw_contours(hu, img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE, masked=1):
    assert hu.shape == img.shape
    size = hu.shape
    hu = hu.astype(int)
    hu = np.reshape(hu, size)
    hu = np.fromiter((x for x in hu.ravel()), dtype=np.uint8).reshape(size)
    cnt = cv2.findContours(hu, mode, method)[0]
    largest_area = 0
    for i in range(len(cnt)):
        area = cv2.contourArea(cnt[i])
        if area > largest_area:
            largest_area = area
            largest_contour = i
    mask = np.zeros(hu.shape[:2], np.uint8)
    cv2.drawContours(mask, cnt, largest_contour, 255, -1)
    if masked:
        thr = mask >= 128
        mask[thr] = img[thr]
    return mask, thr

def show_img(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE, minimum=20):
    a = dict(zip(*np.unique(img, return_counts=True)))
    lst = []
    for i in a.keys():
        if a[i] > minimum and i != 0:
            lst.append(i)
    return max(lst), min(lst)

def output(Input_Image):
    DCM_Img = pydicom.dcmread(Input_Image)
    hu = apply_modality_lut(DCM_Img.pixel_array, DCM_Img)
    mask = convert_hu(hu, 0, 80)
    hu_mask, thr = draw_contours(mask, img=hu) 
    hu_mask = hu_mask.astype(np.float32)
    hu_mask = hu_mask/80*255
    # hu_mask = hu_mask.astype(int)
    # hu_mask = np.dstack([hu_mask, hu_mask, hu_mask])
    # WindowMax, WindowMin = show_img(hu_mask, minimum=20)
    # print(WindowMax, WindowMin)
    # Output_Image = Dicom_to_Image(DCM_Img, 0, 80)
    # Output_Mask, thr = draw_contours(mask, img=Output_Image)
    return hu_mask

def main(path="/content/TransFuse/new_dataset/ISKEMI/test"):
    for f in os.listdir(os.path.join(path, "PNG")):
        print(f)
        Input_Image = os.path.join(path, "PNG", f) 
        DCM_Img = pydicom.dcmread(Input_Image)
        hu = apply_modality_lut(DCM_Img.pixel_array, DCM_Img)
        mask = convert_hu(hu, 0, 100)
        hu_mask, thr = draw_contours(mask, img=hu) 
        WindowMax, WindowMin = show_img(hu_mask, minimum=20)
        print(WindowMax, WindowMin)
        print("\n", "*"*25, "hu_mask", "*"*25, "\n")

        mask_path = os.path.join(path, "MASKS", f.split(".")[0]+".png")
        Output_Image = Dicom_to_Image(DCM_Img, WindowMax, WindowMin)
        img_mask = cv2.imread(mask_path, 0)

        DCM_Mask, thr = draw_contours(mask, img=DCM_Img.pixel_array) 
        Output_Mask, thr = draw_contours(mask, img=Output_Image)
        print(np.unique(Output_Mask))

        img_mask1, thr = draw_contours(img_mask, img=hu) 
        img_mask2, thr = draw_contours(img_mask, img=Output_Image) 

        # show_img(Output_Image, minimum=30)
        # print("\n", "*"*25, "Output_Image", "*"*25, "\n")

        # show_img(Output_Mask, minimum=40)
        # print("\n", "*"*25, "Output_Mask", "*"*25, "\n")

        # show_img(DCM_Mask, minimum=30)
        # print("\n", "*"*25, "DCM_Mask", "*"*25, "\n")

        # show_img(img_mask1, minimum=30)
        # print("\n", "*"*25, "img_mask1", "*"*25, "\n")

        # show_img(img_mask2, minimum=30)
        # print("\n", "*"*25, "img_mask2", "*"*25, "\n")
        
        """modes = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE, cv2.RETR_FLOODFILL]
        methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS]
        for mode in modes:
            for method in methods: 
                if mode == cv2.RETR_FLOODFILL:
                    continue
                hu_mask, thr = draw_contours(mask, hu, mode, method)          
                show_img(hu_mask, mode, method, 20)
                print("\n", "*"*25, "hu_mask", "*"*25, "\n")"""

if __name__ == "__main__":
    main()
