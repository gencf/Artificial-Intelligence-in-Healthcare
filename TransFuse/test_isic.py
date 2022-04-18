import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S
from utils.dataloader import test_dataset
import imageio
import cv2

from IoU import calculateIoU

def mean_iou_np(gt_path, res_path):
    
    groundtruthMask = cv2.imread(gt_path, 0)
    predictedMask = cv2.imread(res_path, 0)    
    shape = groundtruthMask.shape[:2]
    predictedMask = cv2.resize(predictedMask, shape[::-1])
    predictedMask = 255*predictedMask
    iou = calculateIoU(groundtruthMask, predictedMask, showSteps = False)
    return iou

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='npy_files', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default="results", help='path to save inference segmentation')
    parser.add_argument('--epoch', type=int, default=50, help='epoch for inference')
    parser.add_argument('--model_path', type=str, default='models', help='path for testing models')
    parser.add_argument('--png_path', type=str, default='new_dataset/ISKEMI/test/MASKS')
    opt = parser.parse_args()

    ckpt_path = os.path.join(opt.model_path, 'TransFuse_ISKEMI_' + str(opt.epoch) + '_Epoch_best.pth')
    
    model = TransFuse_S().cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', ckpt_path)

    image_root = '{}/data_iskemi_test.npy'.format(opt.test_path)
    gt_root = '{}/mask_iskemi_test.npy'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []

    for i in range(test_loader.size):
        
        image, gt = test_loader.load_data()        
        gt = 1*(gt>0.5)
        image = image.cuda()

        with torch.no_grad():
            _, _, res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)
        
        gt_path = os.path.join(opt.png_path, str(i) + ".png")
        res_path = os.path.join(opt.save_path, str(i) + ".png") 
        
        if opt.save_path is not None:
            cv2.imwrite(res_path, res)
        
        dice = mean_dice_np(gt, res)           
        iou = mean_iou_np(gt_path, res_path)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:f}, Acc: {:.4f}'.
        format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
