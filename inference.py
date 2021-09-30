from numpy.core.fromnumeric import mean
import windowing
from classify import classify
from test_isic import mean_iou_np
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S
from utils.dataloader import test_dataset
import cv2

# PATHS
dcm_path = "DICOM"           # folder contains dicoms
mask_path = "test_dataset/MASKS"          # folder contains masks
resnet_path = "models/second_classification_best.pth"                   # resnet.pth
iskemi_path = "models/iskemi_best.pth"        # iskemi.pth
kanama_path = "models/kanama_best.pth"        # kanama.pth

output_png_path = "results/filtered_images"           # save output pngs (0,80)
kanama_data_path = "results/final_results/KANAMA"    # save classified pngs and masks
iskemi_data_path = "results/final_results/ISKEMI"    # save classified pngs and masks

height = 192    # DO NOT CHANGE THIS VALUE !!!!!!!!!!
width = 256     # DO NOT CHANGE THIS VALUE !!!!!!!!!!

def output(dcm_path, output_png_path):
    os.makedirs(output_png_path, exist_ok=True)
    for i,f in enumerate(os.listdir(dcm_path)):
        png = windowing.output(os.path.join(dcm_path, f)).astype(np.uint8)
        cv2.imwrite(os.path.join(output_png_path, f.split(".")[0]+".png"), png)

def inference(model_path, img): 
    model = TransFuse_S().cuda()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    test_loader = test_dataset(img)   
    image = test_loader.load_data()        
    image = image.cuda()
    with torch.no_grad():
        _, _, res = model(image)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = 1*(res > 0.5)   
    return res
                

if __name__ == "__main__":

    calculate_iou = False

    os.system("rm -rf results")
    output(dcm_path, output_png_path)   
    classify(calculate_iou=calculate_iou, png_path=output_png_path, mask_path=mask_path, iskemi_data_path=iskemi_data_path, kanama_data_path=kanama_data_path, model_path=resnet_path)

    iskemi_iou_results = []
    kanama_iou_results = []

    for i,f in enumerate(os.listdir(os.path.join(iskemi_data_path, "PNG"))):
        img = cv2.imread(os.path.join(iskemi_data_path, "PNG", f))
        img = cv2.resize(img, (width, height))
        result = inference(iskemi_path, img)
        cv2.imwrite(os.path.join(iskemi_data_path, "RESULTS", f), result)

        if calculate_iou:
            iou = mean_iou_np(os.path.join(iskemi_data_path, "MASKS", f), os.path.join(iskemi_data_path, "RESULTS", f))
            if not "IS" in f.split(".")[0]:
                iou = 0
                print("*"*10, "wrong classified image detected", "*"*10)
            print(f, iou)  
            iskemi_iou_results.append(iou)

        shape = cv2.imread(os.path.join(output_png_path, f), 0).shape[:2]
        result = cv2.imread(os.path.join(iskemi_data_path, "RESULTS", f), 0)
        result = cv2.resize(result, shape[::-1])
        cv2.imwrite(os.path.join(iskemi_data_path, "RESULTS", f), result)

    if calculate_iou:
        iskemi_mean_iou = mean(iskemi_iou_results)
        print(f"ISKEMI Mean iou: {iskemi_mean_iou}\n")

    for i,f in enumerate(os.listdir(os.path.join(kanama_data_path, "PNG"))):
        img = cv2.imread(os.path.join(kanama_data_path, "PNG", f))
        img = cv2.resize(img, (width, height))
        result = inference(kanama_path, img)
        cv2.imwrite(os.path.join(kanama_data_path, "RESULTS", f), result)

        if calculate_iou:
            iou = mean_iou_np(os.path.join(kanama_data_path, "MASKS", f), os.path.join(kanama_data_path, "RESULTS", f))
            if not "KA" in f.split(".")[0]:
                iou = 0
                print("*"*10, "wrong classified image detected", "*"*10)
            print(f, iou)
            kanama_iou_results.append(iou)
            
        shape = cv2.imread(os.path.join(output_png_path, f), 0).shape[:2]
        result = cv2.imread(os.path.join(kanama_data_path, "RESULTS", f), 0)
        result = cv2.resize(result, shape[::-1])
        cv2.imwrite(os.path.join(kanama_data_path, "RESULTS", f), result*2)

    if calculate_iou:
        kanama_mean_iou = mean(kanama_iou_results)
        print(f"KANAMA Mean iou: {kanama_mean_iou}\n")
        total_mean_iou = mean(iskemi_iou_results + kanama_iou_results)
        print(f"Total Mean IoU: {total_mean_iou}")
