import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
from glob import glob
import numpy as np
from tqdm import tqdm

import config
from dataset import CXR_Dataset
from validate import valid
from training import train

import pandas as pd
from datetime import datetime

def run(model, model_name):
    
    test_df = pd.read_csv(config.TEST_PATH)
    
    valid_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    test_dataset = CXR_Dataset(dataframe=test_df, transforms=valid_transform)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKER,
        pin_memory=True
    )
    
    model.eval()
    cardio = []
    pneumo = []
    pleural = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            image = batch["img"].to(device=config.DEVICE)
            all_outputs = model(image).cpu()
            cardio.append(torch.sigmoid(all_outputs[:, 0]))
            pneumo.append(torch.sigmoid(all_outputs[:, 1]))
            pleural.append(torch.sigmoid(all_outputs[:, 2]))
    
    cardio = torch.cat(cardio)
    pneumo = torch.cat(pneumo)
    pleural = torch.cat(pleural)

    results_df = pd.DataFrame()
    results_df["Pred_Cardiomegaly"] = cardio.numpy()
    results_df["Pred_Pneumothorax"] = pneumo.numpy()
    results_df["Pred_Pleural"] = pleural.numpy()
    print(results_df)
    print(pleural)
    results_df.to_csv("predictions.csv")
    
if __name__ == "__main__":
    
    model = config.EFFICIENTNETB0
    model.to(device=config.DEVICE)
    model_name = "EFFICIENTNET"
    model.load_state_dict(torch.load("/shared/models/V0/EFFICIENTNETB0/2023-08-20 13:59:55.929073/best_model_config.pth.tar")["model_weights"])
    model.eval()
    run(model, model_name)