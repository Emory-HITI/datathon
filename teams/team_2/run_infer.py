import pandas as pd
import sys   
import importlib
from PIL import Image
from torchvision.transforms import Compose,PILToTensor
import matplotlib.pyplot as plt
from torch2trt import TRTModule
import os
import torch 
sys.path.append("/home/jupyter-medero/kaggle_rsna_breast_cancer/src/roi_det/")
import roi_extract
importlib.reload(roi_extract)
import roi_extract
from torch.utils.data import Dataset,DataLoader 
from tqdm import tqdm 

ATCH_SIZE = 2
# binarization threshold for classification
THRES = 0.31
AUTO_THRES = False
AUTO_THRES_PERCENTILE = 0.97935

# classification model
USE_TRT = True


# roi detection
ROI_YOLOX_INPUT_SIZE = [416, 416]
ROI_YOLOX_CONF_THRES = 0.5
ROI_YOLOX_NMS_THRES = 0.9
ROI_YOLOX_HW = [(52, 52), (26, 26), (13, 13)]
ROI_YOLOX_STRIDES = [8, 16, 32]
ROI_AREA_PCT_THRES = 0.04

# model
MODEL_INPUT_SIZE = [2048, 1024]

MODE = 'LOCAL-VAL'
assert MODE in ['LOCAL-VAL', 'KAGGLE-VAL', 'KAGGLE-TEST']

# settings corresponding to each mode
if MODE == 'KAGGLE-VAL':
    TRT_MODEL_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        f'/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_fold_{i}.pth.tar'
        for i in range(4)
    ]
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/yolox_nano_416_roi_trt_p100.pth'
    CSV_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/_val_fold_0.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/train_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False
elif MODE == 'KAGGLE-TEST':
    TRT_MODEL_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        f'/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_fold_{i}.pth.tar'
        for i in range(4)
    ]
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/yolox_nano_416_roi_trt_p100.pth'
    CSV_PATH = '/kaggle/input/rsna-breast-cancer-detection/test.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/test_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = True
elif MODE == 'LOCAL-VAL':
    TRT_MODEL_PATH = '/home/jupyter-medero/kaggle_rsna_breast_cancer/assets/trained/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        
        f'/home/jupyter-medero/kaggle_rsna_breast_cancer/assets/trained/best_convnext_fold_{i}.pth.tar'
        for i in range(4) #this used to be 4
    ]
    ROI_YOLOX_ENGINE_PATH = '/home/jupyter-medero/kaggle_rsna_breast_cancer/assets/trained/yolox_nano_416_roi_trt.pth'
    #CSV_PATH = '../../datasets/cv/v1/val_fold_0.csv'
    #DCM_ROOT_DIR = '../../datasets/train_images/'
    SAVE_IMG_ROOT_DIR = './temp_save'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False
                              

from torch.nn import functional as F

def resize_and_pad(img, input_size=MODEL_INPUT_SIZE):
    input_h, input_w = input_size
    ori_h, ori_w = img.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    # resize
    img = F.interpolate(img.view(1, 1, ori_h, ori_w),
                        mode="bilinear",
                        scale_factor=ratio,
                        recompute_scale_factor=True)[0, 0]
    # padding
    padded_img = torch.zeros((input_h, input_w),
                             dtype=img.dtype,
                             device='cpu')
    cur_h, cur_w = img.shape
    y_start = (input_h - cur_h) // 2
    x_start = (input_w - cur_w) // 2
    padded_img[y_start:y_start + cur_h, x_start:x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    return padded_img
def min_max_scale(img):
    maxv = img.max()
    minv = img.min()
    if maxv > minv:
        return (img - minv) / (maxv - minv)
    else:
        return img - minv  # ==0
import numpy as np
import torch
from timm.data import resolve_data_config
from timm.models import create_model
from torch import nn


class KFoldEnsembleModel(nn.Module):

    def __init__(self, model_info, ckpt_paths):
        super(KFoldEnsembleModel, self).__init__()
        fmodels = []
        for i, ckpt_path in enumerate(ckpt_paths):
            print(f'Loading model from {ckpt_path}')
            fmodel = create_model(
                model_info['model_name'],
                num_classes=model_info['num_classes'],
                in_chans=model_info['in_chans'],
                pretrained=False,
                checkpoint_path=ckpt_path,
                global_pool=model_info['global_pool'],
            ).eval().to(f'cuda:{i}')
            data_config = resolve_data_config({}, model=fmodel)
            print('Data config:', data_config)
            mean = np.array(data_config['mean']) * 255
            std = np.array(data_config['std']) * 255
            print(f'mean={mean}, std={std}')
            fmodels.append(fmodel)
        self.fmodels = nn.ModuleList(fmodels)

        self.register_buffer('mean',
                             torch.FloatTensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(std).reshape(1, 3, 1, 1))

    def forward(self, x):
        #         x = x.sub(self.mean).div(self.std)

        x = (x - self.mean) / self.std
        probs = []
        for i,fmodel in enumerate(self.fmodels):
            logits = fmodel(x.to(f'cuda:{i}'))
            #             prob = logits.softmax(dim=1)[:, 1]
            prob = logits.sigmoid()[:, 0].to('cpu')
            probs.append(prob)
        probs = torch.stack(probs, dim=1)
        return probs
class MammoDataset(Dataset): 
    def __init__(self,df,transforms): 
        self.df = df 
        self.transforms = transforms
    def __getitem__(self,idx):
        img = Image.open(self.df['png_path'].iloc[idx])
        arr = self.transforms(img)
        return arr,self.df['png_path'].iloc[idx]
    def __len__(self):
        return self.df.shape[0]
def main(): 
    mag_view  = pd.read_csv("/embed/tables/embed_datathon_magview_full.csv",dtype='str') 
    mag_view = mag_view[mag_view['asses'].isin(['N','B','A'])].copy()
    #meta = pd.read_csv("/embed/tables/embed_datathon_metadata_reduced.csv",dtype='str')
    meta = pd.read_csv("final_infer_meta_run2.csv",dtype='str')
    # global vars
    J2K_SUID = '1.2.840.10008.1.2.4.90'
    J2K_HEADER = b"\x00\x00\x00\x0C"
    JLL_SUID = '1.2.840.10008.1.2.4.70'
    JLL_HEADER = b"\xff\xd8\xff\xe0"
    SUID2HEADER = {J2K_SUID: J2K_HEADER, JLL_SUID: JLL_HEADER}
    VOILUT_FUNCS_MAP = {'LINEAR': 0, 'LINEAR_EXACT': 1, 'SIGMOID': 2}
    VOILUT_FUNCS_INV_MAP = {v: k for k, v in VOILUT_FUNCS_MAP.items()}
    roi_extractor = roi_extract.RoiExtractor(engine_path=ROI_YOLOX_ENGINE_PATH,
                                             input_size=ROI_YOLOX_INPUT_SIZE,
                                             num_classes=1,
                                             conf_thres=ROI_YOLOX_CONF_THRES,
                                             nms_thres=ROI_YOLOX_NMS_THRES,
                                             class_agnostic=False,
                                             area_pct_thres=ROI_AREA_PCT_THRES,
                                             hw=ROI_YOLOX_HW,
                                             strides=ROI_YOLOX_STRIDES,
                                             exp=None)
    print('ROI extractor (YOLOX) loaded!')
    model_info = {
    'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
    'num_classes': 1,
    'in_chans': 3,
    'global_pool': 'max',
}
    model = KFoldEnsembleModel(model_info, TORCH_MODEL_CKPT_PATHS)
    model.eval()
    trans = Compose([PILToTensor()])
    mammo_dset = MammoDataset(meta,trans)
    dloader = DataLoader(mammo_dset,num_workers=16,batch_size=1)
    
    out_dict =  dict() 
    counts =0 
    with torch.inference_mode() as f:
        for i,e in enumerate(tqdm(dloader,total=len(dloader))): 
            #inference loop  for croppping code 
            arr,paths = e 
            paths = paths[0]
            arr = arr[0].to(torch.float)
            #arr = trans(x)
            sc = min_max_scale(arr)
            img_yolox = (sc * 255).to(torch.float)
            try: 
                xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox.squeeze(0))
                x0, y0, x1, y1 = xyxy
                crop = arr[0][y0:y1, x0:x1]
            except:
                continue 
            sc_cropped = min_max_scale(crop)
            sc_cropped=  resize_and_pad(sc_cropped)*255
            sc_cropped = sc_cropped.permute(2,0,1)
            outs= model(sc_cropped.unsqueeze(0).to('cpu')).mean(axis=-1)
            out_dict[paths]= float(outs[0])
            if  i %1000==0 and len(out_dict)>0: 
                to_save = meta[meta['png_path'].isin(out_dict.keys())] 
                to_save['score'] = to_save['png_path'].map(out_dict)
                to_save.to_csv(f'/shared/covnext_run3/predicted_batch_{counts}.csv',index=False)
                out_dict = dict()
                counts +=1 
if __name__=='__main__':
    main()
