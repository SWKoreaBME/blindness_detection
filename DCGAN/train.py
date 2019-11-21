from torch.utils.data import DataLoader
from torchvision import utils

from aptos_dataset import aptos_dataset
from preprocessing import preprocessing

data_dir = '../../../Data/APTOS_2019/train_images/'
label_file = '../../../Data/APTOS_2019/train_2019.csv'

dataset = aptos_dataset(d_path=data_dir, label_file=label_file)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for i_batch, sample_batch in enumerate(dataloader):
    
    X_ = sample_batch['image']
    y_true = sample_batch['label']

    # TODO: import Generator
    # TODO: import Classifier