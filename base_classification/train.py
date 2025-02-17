from torch.utils.data import DataLoader
from torchvision import utils
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

from aptos_dataset import aptos_dataset, dataloaders
from preprocessing import preprocessing
# from model import classifier

import time
import copy

data_dir = '/media/sangwook/MGTEC/blindness_detection_data/2019/train_images/'
label_file = '/media/sangwook/MGTEC/blindness_detection_data/2019/train_2019.csv'

dataset = aptos_dataset(d_path=data_dir, da_root_path = './', label_file=label_file, da=False)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 5
num_epochs = 100

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=51200, out_features=num_classes, bias=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# dataloaders
dataloaders, dataset_sizes = dataloaders(dataset, validation_split = 0.2, shuffle_dataset = True, random_seed = 102, batch_size = 8)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for i, batch in enumerate(dataloaders[phase]):
                inputs = batch['image'].to(device).float()
                labels = batch['label'].to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

model_best = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=num_epochs)

torch.save(model_best.state_dict(), './checkpoint/base_classifier_model.pth')