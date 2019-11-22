import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from aptos_dataset import aptos_dataset
from preprocessing import preprocessing

num_classes = 5
running_corrects = 0

model_path = './checkpoint/base_classifier_model.pth'
label_file = '/media/sangwook/MGTEC/blindness_detection_data/2019/train_2019.csv'
data_dir = '/media/sangwook/MGTEC/blindness_detection_data/2019/test_images/'

dataset = aptos_dataset(d_path=data_dir, label_file=label_file)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=51200, out_features=num_classes, bias=False)
model.to(device)
model.load_state_dict(torch.load(model_path))

for i, i_batch in enumerate(dataloader):
    
    labels = i_batch['label'].to(device)
    images = i_batch['image'].to(device).float()
    
    outputs = model(images)
    
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)

# Test accuracy check
test_acc = running_corrects.double() / len(dataloader.dataset)
print(test_acc.data)