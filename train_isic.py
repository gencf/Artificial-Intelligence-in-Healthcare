import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
import shutil
from IPython.display import FileLink


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, best_loss, n, checkpoint, best_iou):
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)

        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.  
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))
                      
    os.makedirs(opt.train_save, exist_ok=True)

    meanloss, meaniou = test(model, opt.test_path)
    if meanloss < best_loss:
        print('mean loss: ', meanloss)
        best_loss = meanloss

    if meaniou > best_iou:
        print('Best IoU: ', meaniou)
        best_iou = meaniou 
        path = os.path.join(opt.train_save, '*best.pth')
        cmd = f"rm {path}"
        os.system(cmd)   
        save_path = os.path.join(opt.train_save, 'TransFuse_ISKEMI_' + str(epoch + n) + '_Epoch_best.pth')
        torch.save(model.state_dict(), save_path)
        print('[Saving Snapshot:]', save_path)  
        
    elif epoch % checkpoint == 0 or epoch == total_step:
        save_path = os.path.join(opt.train_save, 'TransFuse_ISKEMI_' + str(epoch + n) + '_Epoch.pth')
        torch.save(model.state_dict(), save_path)
        print('[Saving Snapshot:]', save_path)  

    return best_loss


def test(model, path):

    model.eval()
    mean_loss = []

    image_root = '{}/data_iskemi_test.npy'.format(path)
    gt_root = '{}/mask_iskemi_test.npy'.format(path)
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    loss_bank = []
    acc_bank = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()
        image = image.cuda()

        with torch.no_grad():
            _, _, res = model(image)
        loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

        res = res.sigmoid().data.cpu().numpy().squeeze()
        gt = 1*(gt>0.5)            
        res = 1*(res > 0.5)

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(iou)
        acc_bank.append(acc)
        
    print('Test Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format(np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

    mean_loss.append(np.mean(loss_bank))

    return mean_loss[0], np.mean(iou_bank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='/kaggle/working/npy_files', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='/kaggle/working/npy_files', help='path to test dataset')
    parser.add_argument('--pretrained_path', type=str,
                        default='/kaggle/input/models', help='path for pretraining')
    parser.add_argument('--train_save', type=str, default='/kaggle/working/TransFuse/snapshots')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--checkpoint', type=int, default=1)

    opt = parser.parse_args()

    # ---- build models ----
    model = TransFuse_S(pretrained=opt.pretrained).cuda()
    n=0
    if opt.pretrained:
        dirlist = [v.split(".")[0].split("_")[-2] for v in os.listdir(opt.pretrained_path)]
        n = max([int(i) for i in dirlist])
        print("ISKEMI", n)
        model_dir = os.path.join(opt.pretrained_path, 'TransFuse_ISKEMI_' + str(n) + '_Epoch.pth')
        model.load_state_dict(torch.load(model_dir))
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = '{}/data_iskemi_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_iskemi_train.npy'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    best_iou = 0
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss, n, opt.checkpoint, best_iou)
        
