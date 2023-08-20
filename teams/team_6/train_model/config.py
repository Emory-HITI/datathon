import torch
import torchvision
import torch.nn as nn

TRAIN_PATH = f'/shared/team6/train_df.csv'
VALID_PATH = f'/shared/team6/val_df.csv'
TEST_PATH = f'/shared/team6/test_df.csv'
DATA_DIR = '/cxr'

LOG_DIR = f'Logs/V0/'
MODEL_DIR = f"models/V0/"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
NUM_WORKER = 30

if torch.cuda.is_available():
    DEVICE = 'cuda:1'
else:
    DEVICE = 'cpu'

    
CLASS_LABELS = {
    'Female': 0,
    'Male': 1
}

EPOCHS = 30

LR = 1e-4
NUM_CLASSES = 3


DENSENET121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
DENSENET121.classifier = nn.Sequential(
        nn.Linear(in_features=(DENSENET121.classifier.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=3)
    )

RESNET50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
RESNET50.fc = nn.Sequential(
        nn.Linear(in_features=(RESNET50.fc.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )

EFFICIENTNETB0 = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
EFFICIENTNETB0.classifier = nn.Sequential(
        nn.Linear(in_features=(EFFICIENTNETB0.classifier[1].in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )