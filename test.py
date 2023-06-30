import glob

import torch
from segformer_pytorch import Segformer
import multiprocessing as mp
import cv2
from test_loader import Data
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.data import DataLoader
from test_schedulers import WarmupCosineLR
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from torch.optim import AdamW, SGD
import torch.nn.functional as F
import torchvision.transforms.functional as F_transform
import torchvision.transforms as T


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


def dice_loss(pred, target, smooth=1.):
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


# if __name__ == "__main__":
#     model = Segformer(
#         dims=(32, 64, 160, 256),      # dimensions of each stage
#         heads=(1, 2, 5, 8),           # heads of each stage
#         ff_expansion=(8, 8, 4, 4),    # feedforward expansion factor of each stage
#         reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
#         num_layers=2,                 # num layers of each stage
#         decoder_dim=256,              # decoder dimension
#         num_classes=2                 # number of segmentation classes
#     )

#     x = torch.randn(1, 3, 64, 64)
#     pred = model(x)  # (1, 4, 64, 64)
#     print(x.size())
#     print(pred.size())


def main(model_save_path):
    num_workers = mp.cpu_count()
    device = torch.device('cpu')

    #root_data = "/Users/kmihara/Downloads/video/*.mp4"
    #root_label = "/Users/kmihara/Downloads/video_label/*.mp4"

    root_data = "./train_xs/*"
    root_label = "./train_ys/*"

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
        video_data = cv2.VideoCapture(data)
        video_label = cv2.VideoCapture(label)
        video_obj_data[data] = video_data
        video_obj_label[label] = video_label
        train_frames[data] = [0, video_label.get(cv2.CAP_PROP_FRAME_COUNT) // 2]
        val_frames[data] = [video_label.get(cv2.CAP_PROP_FRAME_COUNT) // 2,
                            video_label.get(cv2.CAP_PROP_FRAME_COUNT) // 4 * 3]
        test_frames[data] = [video_label.get(cv2.CAP_PROP_FRAME_COUNT) // 4 * 3,
                             video_label.get(cv2.CAP_PROP_FRAME_COUNT)]

    trainset = Data(paths_data, paths_label, video_obj_data, video_obj_label, train_frames)

    model = Segformer(
        dims=(32, 64, 160, 256),      # dimensions of each stage
        heads=(1, 2, 5, 8),           # heads of each stage
        ff_expansion=(8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=2,                 # num layers of each stage
        decoder_dim=256,              # decoder dimension
        num_classes=2                 # number of segmentation classes
    )

    model = model.to(device)

    sampler = RandomSampler(trainset)

    bs = 4
    lr = 1e-3
    epochs = 300

    trainloader = DataLoader(trainset, batch_size=bs, num_workers=0, drop_last=True, pin_memory=True, sampler=sampler)
    #valloader = DataLoader(valset, batch_size=bs, num_workers=0, pin_memory=True)

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

        # for iter, (img, lbl) in pbar:
        # for i in range(len(trainloader)):
        count_iter = 0
        for img, lbl in iter(trainloader):
            count_iter += 1
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)

            with autocast(enabled=False):
                logits = model(img)
                logits = F_transform.resize(img=logits, size=(64, 64), interpolation=T.InterpolationMode.NEAREST)
                loss = loss_fn(logits, lbl)

            loss.backward()
            optimizer.step()
            scheduler.step()
            #torch.cuda.synchronize()

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
                    #torch.cuda.empty_cache()
                    print("early stop...")
                    break

        #train_loss /= (len(trainloader))
        #torch.cuda.empty_cache()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='configs/car.yaml', help='Configuration file to use')
    # args = parser.parse_args()

    # with open(args.cfg) as f:
    #     cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # save_dir = Path(cfg['SAVE_DIR'])
    # save_dir.mkdir(exist_ok=True)
    # main(save_dir)
    main("./model.pt")
