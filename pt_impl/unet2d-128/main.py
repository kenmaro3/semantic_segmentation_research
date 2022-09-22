import torch 
import argparse
import yaml
import time
import sys
sys.path.append("../")
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
import time
from car_video import Car
import glob
import cv2 as cv
import torch.nn.functional as F

from losses import *
from losses import BCE_Dice
from optimizers import *
from schedulers import WarmupCosineLR
#from model_mobile_thin import UNet2D
from mobile_ultra_thin import UNet2D
from torch import nn

from torch.optim import AdamW, SGD

def get_params(model):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    return params


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, metrics=None, bce_weight=0.5):
    # Dice LossとCategorical Cross Entropyを混ぜていい感じにしている
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss


def main(model_save_path):
    num_workers = mp.cpu_count()
    device = torch.device('cuda')

    #root_data = "/Users/kmihara/Downloads/video/*.mp4"
    #root_label = "/Users/kmihara/Downloads/video_label/*.mp4"

    root_data = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video/*"
    root_label = "/home/ubuntu/workdir/nur/semantic-segmentation/data/CAR/training/video_label/*"

    paths_data = sorted(glob.glob(root_data))
    paths_label = sorted(glob.glob(root_label))
    print("list of sequentially data is:  ", paths_data)
    print("list of sequentially label is: ", paths_label)


    video_obj_data = {}
    video_obj_label = {}
    train_frames = {}
    val_frames = {}
    test_frames = {}

    for data, label in zip(paths_data, paths_label):
        video_data = cv.VideoCapture(data)
        video_label = cv.VideoCapture(label)
        video_obj_data[data]=video_data
        video_obj_label[label]=video_label
        train_frames[data] = [0, video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2]
        val_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 2,
                                video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3]
        test_frames[data] = [video_label.get(cv.CAP_PROP_FRAME_COUNT) // 4 * 3,
                                    video_label.get(cv.CAP_PROP_FRAME_COUNT)]


    trainset = Car(paths_data, paths_label, video_obj_data, video_obj_label, train_frames)
    valset = Car(paths_data, paths_label, video_obj_data, video_obj_label, train_frames)

    
    model = UNet2D(3, 1)
    model = model.to(device)

    sampler = RandomSampler(trainset)

    
    bs = 4
    lr = 1e-3
    epochs = 300

    trainloader = DataLoader(trainset, batch_size=bs, num_workers=0, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=bs, num_workers=0, pin_memory=True)

    iters_per_epoch = len(trainset) // bs
    # class_weights = trainset.class_weights.to(device)
    loss_fn = calc_loss
    #loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(get_params(model), lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01)
    scheduler = WarmupCosineLR(optimizer, epochs * iters_per_epoch, 0.9, iters_per_epoch * 0, 0.1)
    # scaler = GradScaler(enabled=False)


    train_loss_list = []
    early_stop_count = 0
    early_stop_threshold = 15

    for epoch in range(epochs):
        print(f"\nepoch loop: {epoch}/{epochs}")
        model.train()

        train_loss = 0.0
        #pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        #for iter, (img, lbl) in pbar:
        #for i in range(len(trainloader)):
        count_iter = 0
        for img, lbl in iter(trainloader):
            count_iter += 1
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)
            
            with autocast(enabled=False):
                logits = model(img)
                loss = loss_fn(logits, lbl)


            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            #lr = sum(lr) / count_iter
            train_loss += loss.item()
        print(f"train_loss: {train_loss}")
        train_loss_list.append(train_loss)

        if len(train_loss_list) == 0:
            torch.save(model, model_save_path)
        elif len(train_loss_list) == 1:
            torch.save(model, model_save_path)
        else:
            if min(train_loss_list[:-1]) > train_loss:
                torch.save(model, model_save_path)
                print("saving model...")
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(f"early stop count: {early_stop_count} / {early_stop_threshold}")
                if early_stop_count >= early_stop_threshold:
                    torch.cuda.empty_cache()
                    print("early stop...")
                    break

        
        #train_loss /= (len(trainloader)) 
        torch.cuda.empty_cache()





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='configs/car.yaml', help='Configuration file to use')
    # args = parser.parse_args()

    # with open(args.cfg) as f:
    #     cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # save_dir = Path(cfg['SAVE_DIR'])
    # save_dir.mkdir(exist_ok=True)
    # main(save_dir)
    main("./model_128_ultra_thin.pt")

