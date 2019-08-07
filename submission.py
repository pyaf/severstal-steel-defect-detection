import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from PIL import Image
from models import get_model
from utils import *
from image_utils import *
from mask_functions import *

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-c", "--ckpt_path",
                        dest="ckpt_path", help="Checkpoint to use")
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = albumentations.Compose(
            [
                albumentations.Rotate(limit=180, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                #albumentations.RandomScale(scale_limit=0.1),
                albumentations.ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=120,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),

            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                #albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return fname, torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_model_name_fold(ckpt_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/ckpt12.pth
    model_folder = ckpt_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2][1:]  # f0 -> 0
    return model_name, int(fold)



def post_process(probability, threshold, min_size):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num



if __name__ == "__main__":
    """
    uses given ckpt to predict on test data and save sigmoid outputs as npy file.
    """
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    predict_on = args.predict_on
    model_name, fold = get_model_name_fold(ckpt_path)
    if predict_on == "test":
        sample_submission_path = "data/sample_submission.csv"
    else:
        sample_submission_path = "data/train.csv"

    sub_path = ckpt_path.replace(".pth", f"{predict_on}.csv")
    npy_path = ckpt_path.replace(".pth", f"{predict_on}%d.npy")
    tta = 0  # number of augs in tta

    root = f"data/{predict_on}_images/"
    size = 1024
    save_npy = False
    save_rle = True
    min_size = 3500
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    use_cuda = True
    num_workers = 2
    batch_size = 4
    device = torch.device("cuda" if use_cuda else "cpu")
    setup(use_cuda)
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes=4)
    model.to(device)
    model.eval()

    print(f"Using {ckpt_path}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"Using tta: {tta}\n")

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    best_threshold = state["best_threshold"]
    best_threshold = 0.5
    print('best_threshold', best_threshold)
    #exit()
    num_batches = len(testset)
    predictions = []
    encoded_pixels = []
    npy_count = 0
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            # images.shape [n, 3, 96, 96] where n is num of 1+tta
            for fnames, images in batch:
                preds = torch.sigmoid(model(images.to(device)))  # [n, num_classes]
                preds = preds.mean(dim=0).detach().tolist()
                predictions.append(preds)
        else:
            fnames, images = batch
            batch_preds = torch.sigmoid(model(images[:, 0].to(device)))
            batch_preds = batch_preds.detach().cpu().numpy()
            if save_npy:
                predictions.extend(batch_preds.tolist())
            if save_rle:
                for fname, preds in zip(fnames, batch_preds):
                    for cls, pred in enumerate(preds):
                        pred, num = post_process(pred, best_threshold, min_size)
                        #pdb.set_trace()
                        rle = mask2rle(pred)
                        name = fname + f"_{cls+1}"
                        predictions.append([name, rle])

        if save_npy:
            if (i+1) % (num_batches//10) == 0:
                print('saving pred npy')
                np.save(npy_path % npy_count, predictions) # raw preds
                npy_count += 1
                del predictions
                predictions = []

    if save_npy:
        np.save(npy_path % npy_count, predictions) # raw preds
        print("Done!")

    if save_rle:
        df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(sub_path, index=False)

