import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from google.colab.patches import cv2_imshow

def loadMask(i):
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    sitkMask = reader.Execute();
    mask = sitk.GetArrayFromImage(sitkMask)
    """

    mask = cv2.imread(i,0)
    mask = cv2.resize(mask, (512, 512))
    mask[mask > 50] = 255
    mask[mask <= 50] = 0
    return mask

def showGrayScale(mask, title):
    cv2_imshow(mask)

# 8 bit ve tek kanallı, normal (0), KANAMAk (1) ve kanama (2) sınıflarından oluşan
# groundTruthMask ve predictedMask'i parametre olarak alip predictedMask'i puanlayan fonksiyon.
# Maskelerin her adımdakı değişiminlerini görmek için showSteps = True.
def calculateIoU(groundTruthMask, predictedMask, showSteps = False):
    
    if showSteps:
        showGrayScale(groundtruthMask, "Ground Truth Maskesi")
        showGrayScale(predictedMask, "Predicted Maskesi")
    
    # 1 ve 2 maskelerini ayrı ayrı oluşturuyoruz (0 ve 255 değerlerinin olduğu birer maske olarak)
    mask1 = np.where(groundtruthMask == 255, 255, 0).astype(np.uint8)
    mask2 = np.where(groundtruthMask == 255, 255, 0).astype(np.uint8)
    

    if showSteps:
        showGrayScale(mask1, "Ground Truth Sınıf 1")
        showGrayScale(mask2, "Ground Truth Sınıf 2")
        

    # İki sınıfı ayrı ayrı dilate ve erode ediyoruz
    kernel = np.ones((3,3))

    erosion1 = cv2.erode(mask1, kernel, iterations=1) 
    dilation1 = cv2.dilate(mask1, kernel, iterations=1)

    erosion2 = cv2.erode(mask2, kernel, iterations=1) 
    dilation2 = cv2.dilate(mask2, kernel, iterations=1)
    
   
    if showSteps:
        showGrayScale(erosion1, "Erode Edilmiş Ground Truth Sınıf 1")
    
    
    # Erode edilmiş ground truth class maskelerini birleştiriyoruz
    erodedGroundtruth = np.zeros(groundtruthMask.shape, dtype = np.uint8)
    erodedGroundtruth[erosion1 == 255] = 255
    erodedGroundtruth[erosion2 == 255] = 255
    
    
    if showSteps:
        showGrayScale(erodedGroundtruth, "Erode Edilmiş Ground Truthların Birleştirilmesi")
    
    
    # Dilate edilmiş ground truth class maskelerini birleştiriyoruz
    dilatedGroundtruth = np.zeros(groundtruthMask.shape, dtype = np.uint8)
    dilatedGroundtruth[dilation1 == 255] = 255
    dilatedGroundtruth[dilation2 == 255] = 255    
    
    
    if showSteps:
        showGrayScale(dilatedGroundtruth, "Dilate Edilmiş Ground Truthların Birleştirilmesi")
    
    
    # Dilate edilmiş ground truth ile kesişim
    intersection = np.where(np.logical_and(dilatedGroundtruth == predictedMask, dilatedGroundtruth != 0), 255, 0)        
    intersectionCount = np.count_nonzero(intersection)

    # Erode edilmiş ground truth ile birleşim
    union = np.where(np.logical_or(erodedGroundtruth != 0, predictedMask != 0), 255, 0)
    unionCount = np.count_nonzero(union)

    score = intersectionCount / unionCount

    
    if showSteps:
        showGrayScale(intersection, "Kesişim")
        showGrayScale(union, "Birleşim")
        print('Kesişim piksel sayısı: ', intersectionCount)
        print('Birleşim piksel sayısı: ', unionCount)
        print('Puan: ', score)
    
    
    return score

if __name__ == "__main__":
    path = "/content/new_dataset/KANAMA/test/MASKS/"
    sum = 0
    for i in range(len(os.listdir(path))):
        groundtruthMask = loadMask(path + str(i)+".png")
        predictedMask = loadMask("/content/results/"+ str(i) +"_pred.jpg")
        iou = calculateIoU(groundtruthMask, predictedMask, showSteps = False)
        iou = str(iou).replace(".", ",")
        print(iou)     