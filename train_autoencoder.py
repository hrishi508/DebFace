import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
# from torchviz import make_dot
from torchvision import transforms

from utils.utils_config import get_config

from backbones.autoencoder import AutoEncoder
# from backbones.am_softmax import Am_softmax
from utils.utils_config import ConfigParams

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1:].to_numpy(dtype='uint8')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label    

def imshow(img):
    # t1 = torch.tensor([0.485, 0.456, 0.406])
    # t2 = torch.tensor([0.229, 0.224, 0.225])
    # img[0]*=t2[0]
    # img[1]*=t2[1]
    # img[2]*=t2[2]

    # img[0]+=t1[0]
    # img[1]+=t1[1]
    # img[2]+=t1[2]

    plt.imshow(np.array(img).transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def getTansform():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])

    return transform

def loadData():
    pass

def train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg):
    size = len(dataloader.dataset)
    # size = 20 # size of dataset
    num_batches = len(dataloader)
    batch_size = cfg.batch_size
    
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(cfg.device)
        X.requires_grad = True

        reconstructed = model(X)

        loss = loss_fn_arr[0](reconstructed, X)
        train_loss += loss.item()
        
        # For visualizing the model
        # make_dot((reconstructed), params=dict(list(model.named_parameters()))).render("AutoEncoder", format="png")        
        
        optimizer.zero_grad()

        # Freeze all encoder (EImg) parameters
        for param in model.encoder.parameters():
            param.requires_grad = False

        loss.backward()
        optimizer.step()

    if cfg.lr_scheduler:
        scheduler.step()

    train_loss /= num_batches
    print(f"\nTraining - Avg loss: {train_loss:>8f} \n")

    train_loss_arr.append(train_loss)

def test(dataloader, model, loss_fn_arr, test_loss_arr, cfg):
    size = len(dataloader.dataset)
    # size = 20 # size of dataset
    num_batches = len(dataloader)
    batch_size = cfg.batch_size
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cfg.device)

            reconstructed = model(X)

            loss = loss_fn_arr[0](reconstructed, X)
            test_loss += loss.item()

    test_loss /= num_batches
    print(f"Testing - Avg loss: {test_loss:>8f} \n")

    test_loss_arr.append(test_loss)

def main(args):

    # get config
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    # create train dataset
    train_data = CustomDataset(cfg.train_dataset_labels, cfg.train_dataset_img_dir, transform=getTansform())

    # visualize train data for debugging
    # img, label = train_data[4888]
    # print(label, type(label))
    # imshow(img)

    # create test split
    train_data, test_data = torch.utils.data.random_split(train_data, [len(train_data) - cfg.val_dataset_size, cfg.val_dataset_size])
    
    # visualize test data for debugging
    # img, label = test_data[500]
    # print(label, type(label))
    # imshow(img)

    # create train dataloader
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

    # visualize train dataloader next image for debugging
    # while True:
    #     tmp = next(iter(train_loader))
    #     imshow(tmp[0][1])
    #     print(tmp[0].shape, type(tmp[0]))
    #     print(tmp[1].shape, type(tmp[1]))

    # create test dataloader
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True)

    # visualize test dataloader next image for debugging
    # while True:
    #     tmp = next(iter(test_loader))
    #     imshow(tmp[0][1])
    #     print(tmp[0].shape, type(tmp[0]))
    #     print(tmp[1].shape, type(tmp[1]))

    model = AutoEncoder(cfg).to(cfg.device)
    # summary(model, (3, 112, 112))

    if cfg.load_weights:
        model.encoder.load_state_dict(torch.load(cfg.model_weights_dir + "encoder.pth"))

    loss_fn_arr = [nn.MSELoss(), nn.L1Loss()]

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    else:
        print("Error while parsing optimizer in config file! Please choose from the supported list of optimizers (sgd or adam) and enter the name correctly in the config file.")
        quit()

    if cfg.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_decay_milestones)

    else:
        scheduler = None

    train_loss_arr = []
    test_loss_arr = []

    # creating a random dataset (same shape as the facial dataset we will be using) for testing the code logic
    # dataloader = []
    # for i in range(2):
    #     X_tmp = torch.randn((10, 3, 112, 112))
    #     # y = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]])
    #     # assuming 4 classes each for gender, age, race and id
    #     y_tmp = torch.randint(2, (10, 4))
    #     dataloader.append((X_tmp, y_tmp))

    epochs = cfg.num_epoch

    try:
        os.makedirs(cfg.model_weights_dir)

    except:
        pass

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg)
        test(test_loader, model, loss_fn_arr, test_loss_arr, cfg)

        # code used for testing model logic using random dataset created above
        # train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg)
        # test(dataloader, model, loss_fn_arr, test_loss_arr, cfg)
        
        if cfg.save_model_weights_every > 0 and (t + 1)%cfg.save_model_weights_every == 0:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S_")
            torch.save(model.decoder.state_dict(), cfg.model_weights_dir + dt_string + f"decoder_weights_epoch_{t+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoEncoder Training in Pytorch")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    main(parser.parse_args())
    print("AutoEncoder Training completed successfully!")

    