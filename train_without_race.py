import os
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

from backbones.debface import DebFaceWithoutRace
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
    # size = 100 # size of dataset
    num_batches = len(dataloader)
    batch_size = cfg.batch_size
    
    model.train()
    train_loss = 0
    correct_G, correct_A, correct_ID, correct_Distr = 0, 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(cfg.device)
        X.requires_grad = True

        out_G1, out_G2, out_G3, out_A1, out_A2, out_A3, out_ID1, out_ID2, out_ID3, out_Distr1, out_Distr2 = model(X)

        y_G1 = y[:, 0].clone()
        y_A1 = torch.full(y_G1.shape, 1)
        y_ID1 = torch.full(y_G1.shape, 1)

        y_A2  = y[:, 1].clone()
        y_G2  = torch.full(y_A2.shape, 1)
        y_ID2 = torch.full(y_A2.shape, 1)

        y_ID3  = y[:, 2].clone()
        y_G3  = torch.full(y_ID3.shape, 1)
        y_A3 = torch.full(y_ID3.shape, 1)

        y_Distr11 = torch.tensor([1 for i in range(batch_size)])
        y_Distr12 = torch.tensor([1 for i in range(batch_size)])

        y_Distr21 = torch.tensor([0 for i in range(batch_size)])
        y_Distr22 = torch.tensor([1 for i in range(batch_size)])

        # Classification losses
        loss_G1 = loss_fn_arr[0](out_G1, y_G1)
        loss_A2 = loss_fn_arr[0](out_A2, y_A2)
        loss_ID3 = loss_fn_arr[0](out_ID3, y_ID3)
        loss_Distr11 = loss_fn_arr[0](out_Distr1, y_Distr11)
        loss_Distr21 = loss_fn_arr[0](out_Distr2, y_Distr21)

        classification_loss = loss_G1 + loss_A2 + loss_ID3 + loss_Distr11 + loss_Distr21
        train_loss += classification_loss.item()

        # Adversarial losses
        loss_A1 = loss_fn_arr[1](out_A1, y_A1)
        loss_ID1 = loss_fn_arr[1](out_ID1, y_ID1)

        loss_G2 = loss_fn_arr[2](out_G2, y_G2)
        loss_ID2 = loss_fn_arr[2](out_ID2, y_ID2)

        loss_G3 = loss_fn_arr[3](out_G3, y_G3)
        loss_A3 = loss_fn_arr[3](out_A3, y_A3)

        loss_Distr12 = loss_fn_arr[4](out_Distr1, y_Distr12)
        loss_Distr22 = loss_fn_arr[4](out_Distr2, y_Distr22)

        adversarial_loss = loss_G2 + loss_G3 + loss_A1 + loss_A3 + loss_ID1 + loss_ID2 + loss_Distr12 + loss_Distr22
        train_loss += adversarial_loss.item()

        # Calculate classifier accuracies and total loss per batch
        with torch.no_grad():
            correct_G += (out_G1.argmax(1) == y_G1).type(torch.float).sum().item()
            correct_A += (out_A2.argmax(1) == y_A2).type(torch.float).sum().item()
            correct_ID += (out_ID3.argmax(1) == y_ID3).type(torch.float).sum().item()            
            correct_Distr += (out_Distr1.argmax(1) == y_Distr11).type(torch.float).sum().item()           
            correct_Distr += (out_Distr2.argmax(1) == y_Distr21).type(torch.float).sum().item()
            
            # if batch!=0 and batch % 10 == 0:
            loss, current = (classification_loss.item() + adversarial_loss.item()), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
        # For visualizing the model
        # make_dot((out_G1, out_G2, out_G3, out_A1, out_A2, out_A3, out_ID1, out_ID2, out_ID3, out_Distr1, out_Distr2), params=dict(list(model.named_parameters()))).render("DebFace_Final_without_race", format="png")        
        
        optimizer.zero_grad()

        # Freeze all model parameters except encoder (EImg) parameters
        for param in model.parameters():
            param.requires_grad = False

        for param in model.encoder.parameters():
            param.requires_grad = True

        # Calculate gradients only for encoder (EImg) parameters
        adversarial_loss.backward(retain_graph=True)

        # Unfreeze all model parameters
        for param in model.parameters():
            param.requires_grad = True

        classification_loss.backward()
        optimizer.step()

    if cfg.lr_scheduler:
        scheduler.step()
    train_loss /= num_batches
    correct_G /= size
    correct_A /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"\nTraining - Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    train_loss_arr.append(train_loss)

def test(dataloader, model, loss_fn_arr, test_loss_arr, cfg):
    size = len(dataloader.dataset)
    # size = 100 # size of dataset
    num_batches = len(dataloader)
    batch_size = cfg.batch_size
    test_loss = 0
    correct_G, correct_A, correct_ID, correct_Distr = 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cfg.device)

            out_G1, out_G2, out_G3, out_A1, out_A2, out_A3, out_ID1, out_ID2, out_ID3, out_Distr1, out_Distr2 = model(X)

            y_G1 = y[:, 0].clone()
            y_A1 = torch.full(y_G1.shape, 1)
            y_ID1 = torch.full(y_G1.shape, 1)

            y_A2  = y[:, 1].clone()
            y_G2  = torch.full(y_A2.shape, 1)
            y_ID2 = torch.full(y_A2.shape, 1)

            y_ID3  = y[:, 2].clone()
            y_G3  = torch.full(y_ID3.shape, 1)
            y_A3 = torch.full(y_ID3.shape, 1)

            y_Distr11 = torch.tensor([1 for i in range(batch_size)])
            y_Distr12 = torch.tensor([1 for i in range(batch_size)])

            y_Distr21 = torch.tensor([0 for i in range(batch_size)])
            y_Distr22 = torch.tensor([1 for i in range(batch_size)])

            # Classification losses
            loss_G1 = loss_fn_arr[0](out_G1, y_G1)
            loss_A2 = loss_fn_arr[0](out_A2, y_A2)
            loss_ID3 = loss_fn_arr[0](out_ID3, y_ID3)
            loss_Distr11 = loss_fn_arr[0](out_Distr1, y_Distr11)
            loss_Distr21 = loss_fn_arr[0](out_Distr2, y_Distr21)

            classification_loss = loss_G1 + loss_A2 + loss_ID3 + loss_Distr11 + loss_Distr21
            test_loss += classification_loss.item()

            # Adversarial losses
            loss_A1 = loss_fn_arr[1](out_A1, y_A1)
            loss_ID1 = loss_fn_arr[1](out_ID1, y_ID1)

            loss_G2 = loss_fn_arr[2](out_G2, y_G2)
            loss_ID2 = loss_fn_arr[2](out_ID2, y_ID2)

            loss_G3 = loss_fn_arr[3](out_G3, y_G3)
            loss_A3 = loss_fn_arr[3](out_A3, y_A3)

            loss_Distr12 = loss_fn_arr[4](out_Distr1, y_Distr12)
            loss_Distr22 = loss_fn_arr[4](out_Distr2, y_Distr22)

            adversarial_loss = loss_G2 + loss_G3 + loss_A1 + loss_A3 + loss_ID1 + loss_ID2 + loss_Distr12 + loss_Distr22
            test_loss += adversarial_loss.item()

            # Calculate classifier accuracies
            correct_G += (out_G1.argmax(1) == y_G1).type(torch.float).sum().item()
            correct_A += (out_A2.argmax(1) == y_A2).type(torch.float).sum().item()
            correct_ID += (out_ID3.argmax(1) == y_ID3).type(torch.float).sum().item()            
            correct_Distr += (out_Distr1.argmax(1) == y_Distr11).type(torch.float).sum().item()           
            correct_Distr += (out_Distr2.argmax(1) == y_Distr21).type(torch.float).sum().item()

    test_loss /= num_batches
    correct_G /= size
    correct_A /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"Testing - Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

    model = DebFaceWithoutRace(cfg).to(cfg.device)
    # summary(model, (3, 112, 112))

    weight_G = torch.tensor([(1/cfg.n_gender_classes) for i in range(cfg.n_gender_classes)])
    weight_A = torch.tensor([(1/cfg.n_age_classes) for i in range(cfg.n_age_classes)])
    weight_ID = torch.tensor([(1/cfg.n_id_classes) for i in range(cfg.n_id_classes)])
    weight_Distr = torch.tensor([(1/cfg.n_distr_classes) for i in range(cfg.n_distr_classes)])

    loss_fn_arr = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=weight_G), nn.CrossEntropyLoss(weight=weight_A), nn.CrossEntropyLoss(weight=weight_ID), nn.CrossEntropyLoss(weight=weight_Distr)]

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
    # for i in range(10):
    #     X_tmp = torch.randn((10, 3, 112, 112))
    #     # y = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    #     # assuming 4 classes each for gender, age and id
    #     y_tmp = torch.randint(2, (10, 3))
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
        # train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, cfg)
        # test(dataloader, model, loss_fn_arr, test_loss_arr, cfg)

        if cfg.save_model_weights_every > 0 and (t + 1)%cfg.save_model_weights_every == 0:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S_")
            torch.save(model.state_dict(), cfg.model_weights_dir + dt_string + "weights.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DebFace Training in Pytorch")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    main(parser.parse_args())
    print("DebFace Training completed successfully!")

    