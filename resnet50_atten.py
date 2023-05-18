# Installs 

!pip install mlflow==1.30.1 dagshub==0.2.12 progressbar2==4.2.0 GPUtil==1.4.0 albumentations==1.3.0

# Imports

import albumentations as A
import cv2
from dataclasses import dataclass, asdict
import dagshub
from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
import mlflow
from numba import cuda
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint
import progressbar
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision import models, transforms, utils
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch import flatten
from torchmetrics import MeanSquaredError

# General utilities 

def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True
    
def free_gpu_cache():
    print("Initial GPU usage")
    gpu_usage()                             

    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU usage after emptying the cache")
    gpu_usage()


# Set up

set_seeds()

DIR_PATH = "/kaggle/input/facial-keypoints-detection/"
training_data = pd.read_csv(f"{DIR_PATH}training.zip")
test_data = pd.read_csv(f"{DIR_PATH}test.zip")
id_lookup_table = pd.read_csv(f"{DIR_PATH}IdLookupTable.csv")

@dataclass  
class ModelParams:
    BATCH_SIZE: int = 64
    VALID_SIZE: float = 0.1
    N_EPOCHS: int = 170
    IMG_SIZE: int = 96
    OUTPUT_SIZE: int = 30 
    S_OUTPUT_SIZE: int = 8
    L_OUTPUT_SIZE: int = 22
    LEARNING_RATE: float = 0.001

# dagshub.init("facial_reg_model", "caddis90", mlflow=True)
# mlflow.set_tracking_uri('https://dagshub.com/caddis90/facial_reg_model.mlflow')
mlflow.set_experiment(experiment_name="cnn")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

l_dataset_cols = [
    'left_eye_inner_corner_x','left_eye_inner_corner_y', 'left_eye_outer_corner_x',
    'left_eye_outer_corner_y', 'right_eye_inner_corner_x','right_eye_inner_corner_y',
    'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
    'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
    'right_eyebrow_outer_end_y', 'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
    'mouth_center_top_lip_y', 'Image']

s_dataset_cols = [
    'left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
    'right_eye_center_y','nose_tip_x', 'nose_tip_y',
    'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y','Image']

l_dataset = training_data[l_dataset_cols]
s_dataset = training_data[s_dataset_cols]



import torch.nn as nn
import torch
import math


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class ResNet(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers, num_classes, model_path="model.pkl"):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)


        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self.se1 = SE_Block(512 * Bottleneck.expansion)
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)


        x = self.se1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Data 

class FacialKeypointsDataset(Dataset):
    
    def __init__(self, dataset, train=True, transform=None):
        self.dataset = dataset
        self.train = train
        self.transform = transform

    def get_image(self, idx):
        image = np.fromstring(self.dataset.iloc[idx, -1], sep=' ', dtype = np.uint8)
        image = image.astype(np.float32)
        image = image.reshape(ModelParams.IMG_SIZE, ModelParams.IMG_SIZE, 1) 
        
        return image
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):        
        if self.train:
            keypoints = self.dataset.iloc[idx, :-1].values.astype(np.float32)
            total_keypoints = int(len(keypoints)/2)
            keypoints = keypoints.reshape([total_keypoints, 2])
        else:
            keypoints = None
        
        if self.transform:
            data_cols = self.dataset.columns.tolist()
            sample = self.transform(
                image=self.get_image(idx), 
                keypoints=keypoints, 
                class_labels=data_cols[0:-1])
            sample["keypoints"] = torch.tensor(list(sum(sample["keypoints"], ()))).float()
        else:
            sample = {"image": self.get_image(idx)}
            
        sample["image"] = torch.from_numpy(sample["image"].transpose(2, 0, 1)).float()
        sample["image"] = sample["image"] / 255
        
        return sample

def prepare_dataloaders(dataset, valid_size, batch_size):
    dataset_len = len(dataset)
    dataset_indices = list(range(dataset_len))
    np.random.shuffle(dataset_indices)
    split = int(np.floor(valid_size * dataset_len))
    train_idx, valid_idx = dataset_indices[split:], dataset_indices[:split]
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    
    return train_loader, valid_loader


# Utilities

def show_image(image, training_key_points, test_key_points=[]):  
    image = image.numpy().transpose(1, 2, 0)

    plt.imshow(image, cmap="gray")

    total_keypoints = int(len(training_key_points)/2)
    training_key_points = training_key_points.reshape([total_keypoints, 2])
    plt.plot(training_key_points[:,0], training_key_points[:,1], 'gx')
    
    if len(test_key_points) > 0:        
        test_key_points = test_key_points.reshape([total_keypoints, 2])
        plt.plot(test_key_points[:,0], test_key_points[:,1], 'rx')

        
def mask_output_and_target(output, target):
    mask = torch.isnan(target)
    return output[~mask], target[~mask]

def train(train_loader, valid_loader, model, optimizer, scheduler):
    with mlflow.start_run():
        mlflow.log_params(asdict(ModelParams()))
        for epoch in progressbar.progressbar(range(ModelParams.N_EPOCHS)):
            epoch_train_loss, epoch_valid_loss = 0.0, 0.0

            model.train() 
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(batch['image'].to(device))
                output_with_mask, target_with_mask = mask_output_and_target(
                        output=output, 
                        target=batch['keypoints']
                )
                loss = criterion(output_with_mask, target_with_mask.to(device))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()*batch['image'].size(0)
                batch_train_rmse = rmse(output_with_mask.cpu(), target_with_mask.cpu())

            epoch_train_rmse = rmse.compute()
            rmse.reset()

            with torch.no_grad():
                model.eval() 
                for i, batch in enumerate(valid_loader):
                    output = model(batch['image'].to(device))
                    output_with_mask, target_with_mask = mask_output_and_target(
                            output=output, 
                        target=batch['keypoints']
                    )
                    loss = criterion(output_with_mask, target_with_mask.to(device))
                    epoch_valid_loss += loss.item()*batch['image'].size(0)
                    batch_valid_rmse = rmse(output_with_mask.cpu(), target_with_mask.cpu())

                epoch_valid_rmse = rmse.compute()
                rmse.reset()
                epoch_train_loss = np.sqrt(epoch_train_loss/len(train_loader.sampler.indices))
                epoch_valid_loss = np.sqrt(epoch_valid_loss/len(valid_loader.sampler.indices))
                
                mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
                mlflow.log_metric("valid_loss", epoch_valid_loss, step=epoch)
                mlflow.log_metric("train_rmse", epoch_train_rmse, step=epoch)
                mlflow.log_metric("valid_rmse", epoch_valid_rmse, step=epoch)
            
            scheduler.step(metrics=epoch_valid_loss)

            
def predict(model, test_loader):    
    model.eval()
    with torch.no_grad():
        for i, batch in progressbar.progressbar(enumerate(test_loader)):
            output = model(batch['image'].to(device)).cpu().numpy()
            output = np.clip(output, a_min=0, a_max=ModelParams.IMG_SIZE)
            if i == 0:
                test_predictions = output
            else:
                test_predictions = np.vstack((test_predictions, output))
        
    return test_predictions

def create_submission(predictions, prediction_features, id_lookup_table=id_lookup_table):
    features = list(id_lookup_table['FeatureName'])
    img_ids = list(id_lookup_table['ImageId']-1) 

    prediction_indices = [prediction_features.index(feature) for feature in features]

    submission = pd.DataFrame({
        "RowId": list(id_lookup_table['RowId']),
        "Location": [predictions[x][y] for x, y in zip(img_ids, prediction_indices)]
    })
    submission.to_csv("submission.csv",index = False)
    print("Submission successful!")
       
# Model

l_resnet50 = resnet50(num_classes = ModelParams.L_OUTPUT_SIZE)

s_resnet50 = resnet50(num_classes = ModelParams.S_OUTPUT_SIZE)

transformations = A.Compose([
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
    A.Affine(p=0.2),
#     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=False, p=0.2),
    A.OneOf([
#             A.GaussNoise(p=0.8),
#             A.RandomGamma(p=0.8),
            A.Blur(blur_limit=3, p=0.8),
            A.PixelDropout(p=0.8)
        ], p=1.0),     
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
    ],
    keypoint_params = A.KeypointParams(format = 'xy', remove_invisible=False)
)

s_trainset = FacialKeypointsDataset(s_dataset, transform=transformations)
l_trainset = FacialKeypointsDataset(l_dataset, transform=transformations)
testset = FacialKeypointsDataset(test_data, train=False)

s_train_loader, s_valid_loader = prepare_dataloaders(
        s_trainset, 
        valid_size=ModelParams.VALID_SIZE, 
        batch_size=ModelParams.BATCH_SIZE
)
l_train_loader, l_valid_loader = prepare_dataloaders(
        l_trainset, 
        valid_size=ModelParams.VALID_SIZE, 
        batch_size=ModelParams.BATCH_SIZE
)
test_loader = DataLoader(testset, batch_size=ModelParams.BATCH_SIZE)

l_model = l_resnet50
l_model = l_model.to(device)
s_model = s_resnet50
s_model = s_model.to(device)

criterion = nn.MSELoss().to(device)
rmse = MeanSquaredError(squared=False).to(device)

l_optimizer = optim.Adam(l_model.parameters(), lr=ModelParams.LEARNING_RATE)
l_scheduler = ReduceLROnPlateau(
    optimizer=l_optimizer, 
    mode="min", 
    factor=0.5,
    patience=5,
    min_lr=1e-15
)

s_optimizer = optim.Adam(s_model.parameters(), lr=ModelParams.LEARNING_RATE)
s_scheduler = ReduceLROnPlateau(
    optimizer=s_optimizer, 
    mode="min", 
    factor=0.5,
    patience=5,
    min_lr=1e-15
)



train(
    train_loader=l_train_loader, 
    valid_loader=l_valid_loader, 
    model=l_model, 
    optimizer=l_optimizer, 
    scheduler=l_scheduler
)

train(
    train_loader=s_train_loader, 
    valid_loader=s_valid_loader, 
    model=s_model, 
    optimizer=s_optimizer, 
    scheduler=s_scheduler
)


l_predictions = predict(model=l_model, test_loader=test_loader)
s_predictions = predict(model=s_model, test_loader=test_loader)
predictions = np.hstack((l_predictions, s_predictions))
prediction_features = l_dataset_cols[:-1] + s_dataset_cols[:-1]
create_submission(predictions=predictions, prediction_features=prediction_features)
l_predictions = predict(model=l_model, test_loader=test_loader)
s_predictions = predict(model=s_model, test_loader=test_loader)
predictions = np.hstack((l_predictions, s_predictions))
prediction_features = l_dataset_cols[:-1] + s_dataset_cols[:-1]
create_submission(predictions=predictions, prediction_features=prediction_features)