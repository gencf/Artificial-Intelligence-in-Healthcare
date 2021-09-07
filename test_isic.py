import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S
from utils.dataloader import test_dataset
import imageio

def mean_iou_np(groundTruthMask, predictedMask):
    
    # 1 ve 2 maskelerini ayrı ayrı oluşturuyoruz (0 ve 255 değerlerinin olduğu birer maske olarak)
    mask1 = np.where(groundtruthMask == 255, 255, 0).astype(np.uint8)
    mask2 = np.where(groundtruthMask == 255, 255, 0).astype(np.uint8)      

    # İki sınıfı ayrı ayrı dilate ve erode ediyoruz
    kernel = np.ones((3,3))

    erosion1 = cv2.erode(mask1, kernel, iterations=1) 
    dilation1 = cv2.dilate(mask1, kernel, iterations=1)

    erosion2 = cv2.erode(mask2, kernel, iterations=1) 
    dilation2 = cv2.dilate(mask2, kernel, iterations=1)
        
    # Erode edilmiş ground truth class maskelerini birleştiriyoruz
    erodedGroundtruth = np.zeros(groundtruthMask.shape, dtype = np.uint8)
    erodedGroundtruth[erosion1 == 255] = 255
    erodedGroundtruth[erosion2 == 255] = 255
    
    # Dilate edilmiş ground truth class maskelerini birleştiriyoruz
    dilatedGroundtruth = np.zeros(groundtruthMask.shape, dtype = np.uint8)
    dilatedGroundtruth[dilation1 == 255] = 255
    dilatedGroundtruth[dilation2 == 255] = 255    
       
    # Dilate edilmiş ground truth ile kesişim
    intersection = np.where(np.logical_and(dilatedGroundtruth == predictedMask, dilatedGroundtruth != 0), 255, 0)        
    intersectionCount = np.count_nonzero(intersection)

    # Erode edilmiş ground truth ile birleşim
    union = np.where(np.logical_or(erodedGroundtruth != 0, predictedMask != 0), 255, 0)
    unionCount = np.count_nonzero(union)

    score = intersectionCount / unionCount 
    return score


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
    parser.add_argument('--test_path', type=str,
                        default='/kaggle/working/npy_files', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default="/kaggle/working/results", help='path to save inference segmentation')
    parser.add_argument('--epoch', type=int, default=50, help='epoch for inference')
    parser.add_argument('--model_path', type=str, default='/kaggle/working/models', help='path for testing models')
    
    opt = parser.parse_args()

    ckpt_path = os.path.join(opt.model_path, 'TransFuse_ISKEMI_' + str(opt.epoch) + '_Epoch.pth')
    
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
        iou = mean_iou_np(gt, res)

        gt = 1*(gt>0.5)
        image = image.cuda()

        with torch.no_grad():
            _, _, res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)

        if opt.save_path is not None:
            imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', res)
            imageio.imwrite(opt.save_path+'/'+str(i)+'_gt.jpg', gt)

        dice = mean_dice_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
