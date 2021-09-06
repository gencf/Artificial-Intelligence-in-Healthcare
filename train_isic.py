import os
import torch
import argparse
import yaml
import tqdm
import sys
import shutil
import argparse

from torch.autograd import Variable
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.dataloader import *
from utils.utils import *


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xomcnet.yaml')
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="new_dataset")
    parser.add_argument('--inme', type=str, default="ISKEMI")
    parser.add_argument('--dicom', type=int, default=0)
    return parser.parse_args()

def train(opt, args):
    dataset = args.dataset
    path = os.path.join("/content/drive/MyDrive/İNAN/SağlıktaYapayZeka/UACANet", dataset, args.inme)
    n = 0
    model = eval(opt.Model.name)(opt.Model).cuda()

    if args.pretrained:
        dirlist = [v.split(".")[0].split("_")[-2] for v in os.listdir(path)]
        n = max([int(i) for i in dirlist])
        model.load_state_dict(torch.load(os.path.join(path, "UACANet_" + args.inme + "_" + str(n) + "_Epoch.pth")))
        model.cuda()
        model.eval()

    image_root = os.path.join(opt.Train.train_path, args.inme, "train", 'PNG')
    gt_root = os.path.join(opt.Train.train_path, args.inme, "train", 'MASKS')
    print(args.inme, n)
    train_dataset = PolypDataset(image_root, gt_root, opt.Train)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=opt.Train.batchsize,
                                   shuffle=opt.Train.shuffle,
                                   num_workers=opt.Train.num_workers,
                                   pin_memory=opt.Train.pin_memory)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.Train.lr)
    scheduler = PolyLr(optimizer, gamma=opt.Train.gamma,
                       minimum_lr=opt.Train.min_learning_rate,
                       max_iteration=len(train_loader) * opt.Train.epoch,
                       warmup_iteration=opt.Train.warmup_iteration)
    model.train()

    print('#' * 20, 'Train prep done, start training', '#' * 20)

    for epoch in tqdm.tqdm(range(1, opt.Train.epoch + 1), desc='Epoch', total=opt.Train.epoch, position=0,
                           bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}'):
        pbar = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(train_loader), position=1,
                         leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        for i, sample in pbar:
            optimizer.zero_grad()
            images, gts = sample['image'], sample['gt']
            images = images.cuda()
            gts = gts.cuda()
            out = model(images, gts)
            out['loss'].backward()
            clip_gradient(optimizer, opt.Train.clip)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': out['loss'].item()})

        os.makedirs(opt.Train.train_save, exist_ok=True)
        if epoch % opt.Train.checkpoint_epoch == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.Train.train_save, "UACANet_" + args.inme + "_" + str(epoch + n) + '_Epoch.pth'))
            shutil.copy("/content/UACANet/snapshots/UACANet-L/" + "UACANet_" + args.inme + "_" + str(epoch + n) + "_Epoch.pth",
                        os.path.join(path, "UACANet_" + args.inme + "_" + str(epoch + n) + "_Epoch.pth"))

    print('#' * 20, 'Train done', '#' * 20)


if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    if args.dicom:
        opt.Train.transforms.resize.size = [352, 352, 3]
    train(opt, args)
