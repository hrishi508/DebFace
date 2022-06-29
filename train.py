import os
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchsummary import summary
# from torchviz import make_dot

from utils.utils_config import get_config

from backbones.debface import DebFace
# from backbones.am_softmax import Am_softmax
from utils.utils_config import ConfigParams

def train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg):
    # size = len(dataloader.dataset)
    size = 20 # size of dataset
    num_batches = len(dataloader)
    batch_size = int(size/num_batches)
    
    model.train()
    train_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(cfg.device)
        y = y.to(cfg.device)
        X.requires_grad = True

        out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2 = model(X)

        y_G1 = y[:, 0].clone()
        y_A1 = torch.full(y_G1.shape, 1).to(cfg.device)
        y_R1 = torch.full(y_G1.shape, 1).to(cfg.device)
        y_ID1 = torch.full(y_G1.shape, 1).to(cfg.device)

        y_A2  = y[:, 1].clone()
        y_G2  = torch.full(y_A2.shape, 1).to(cfg.device)
        y_R2  = torch.full(y_A2.shape, 1).to(cfg.device)
        y_ID2 = torch.full(y_A2.shape, 1).to(cfg.device)

        y_R3  = y[:, 2].clone()
        y_G3  = torch.full(y_R3.shape, 1).to(cfg.device)
        y_A3  = torch.full(y_R3.shape, 1).to(cfg.device)
        y_ID3 = torch.full(y_R3.shape, 1).to(cfg.device)

        y_ID4  = y[:, 3].clone()
        y_G4  = torch.full(y_ID4.shape, 1).to(cfg.device)
        y_A4  = torch.full(y_ID4.shape, 1).to(cfg.device)
        y_R4 = torch.full(y_ID4.shape, 1).to(cfg.device)

        y_Distr11 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)
        y_Distr12 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)

        y_Distr21 = torch.tensor([0 for i in range(batch_size)]).to(cfg.device)
        y_Distr22 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)

        # Classification losses
        loss_G1 = loss_fn_arr[0](out_G1, y_G1)
        loss_A2 = loss_fn_arr[0](out_A2, y_A2)
        loss_R3 = loss_fn_arr[0](out_R3, y_R3)
        loss_ID4 = loss_fn_arr[0](out_ID4, y_ID4)
        loss_Distr11 = loss_fn_arr[0](out_Distr1, y_Distr11)
        loss_Distr21 = loss_fn_arr[0](out_Distr2, y_Distr21)

        classification_loss = loss_G1 + loss_A2 + loss_R3 + loss_ID4 + loss_Distr11 + loss_Distr21
        train_loss += classification_loss.item()

        # Adversarial losses
        loss_A1 = loss_fn_arr[1](out_A1, y_A1)
        loss_R1 = loss_fn_arr[1](out_R1, y_R1)
        loss_ID1 = loss_fn_arr[1](out_ID1, y_ID1)
        
        loss_G2 = loss_fn_arr[2](out_G2, y_G2)
        loss_R2 = loss_fn_arr[2](out_R2, y_R2)
        loss_ID2 = loss_fn_arr[2](out_ID2, y_ID2)

        loss_G3 = loss_fn_arr[3](out_G3, y_G3)
        loss_A3 = loss_fn_arr[3](out_A3, y_A3)
        loss_ID3 = loss_fn_arr[3](out_ID3, y_ID3)

        loss_G4 = loss_fn_arr[4](out_G4, y_G4)
        loss_A4 = loss_fn_arr[4](out_A4, y_A4)
        loss_R4 = loss_fn_arr[4](out_R4, y_R4)

        loss_Distr12 = loss_fn_arr[5](out_Distr1, y_Distr12)
        loss_Distr22 = loss_fn_arr[5](out_Distr2, y_Distr22)

        adversarial_loss = loss_G2 + loss_G3 + loss_G4 + loss_A1 + loss_A3 + loss_A4 + loss_R1 + loss_R2 + loss_R4 + loss_ID1 + loss_ID2 + loss_ID3 + loss_Distr12 + loss_Distr22
        train_loss += adversarial_loss.item()

        # Calculate classifier accuracies and total loss per batch
        with torch.no_grad():
            correct_G += (out_G1.argmax(1) == y_G1).type(torch.float).sum().item()
            correct_A += (out_A2.argmax(1) == y_A2).type(torch.float).sum().item()
            correct_R += (out_R3.argmax(1) == y_R3).type(torch.float).sum().item()
            correct_ID += (out_ID4.argmax(1) == y_ID4).type(torch.float).sum().item()            
            correct_Distr += (out_Distr1.argmax(1) == y_Distr11).type(torch.float).sum().item()           
            correct_Distr += (out_Distr2.argmax(1) == y_Distr21).type(torch.float).sum().item()
            
            # if batch!=0 and batch % 10 == 0:
            # loss, current = (classification_loss.item() + adversarial_loss.item()), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
        # For visualizing the model
        # make_dot((out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2), params=dict(list(model.named_parameters()))).render("DebFace_Final", format="png")

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
    correct_R /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"\nTraining - Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_R: {(100*correct_R):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    train_loss_arr.append(train_loss)

def test(dataloader, model, loss_fn_arr, test_loss_arr, cfg):
    # size = len(dataloader.dataset)
    size = 20 # size of dataset
    num_batches = len(dataloader)
    batch_size = int(size/num_batches)
    test_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cfg.device)
            y = y.to(cfg.device)

            out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2 = model(X)

            y_G1 = y[:, 0].clone()
            y_A1 = torch.full(y_G1.shape, 1).to(cfg.device)
            y_R1 = torch.full(y_G1.shape, 1).to(cfg.device)
            y_ID1 = torch.full(y_G1.shape, 1).to(cfg.device)

            y_A2  = y[:, 1].clone()
            y_G2  = torch.full(y_A2.shape, 1).to(cfg.device)
            y_R2  = torch.full(y_A2.shape, 1).to(cfg.device)
            y_ID2 = torch.full(y_A2.shape, 1).to(cfg.device)

            y_R3  = y[:, 2].clone()
            y_G3  = torch.full(y_R3.shape, 1).to(cfg.device)
            y_A3  = torch.full(y_R3.shape, 1).to(cfg.device)
            y_ID3 = torch.full(y_R3.shape, 1).to(cfg.device)

            y_ID4  = y[:, 3].clone()
            y_G4  = torch.full(y_ID4.shape, 1).to(cfg.device)
            y_A4  = torch.full(y_ID4.shape, 1).to(cfg.device)
            y_R4 = torch.full(y_ID4.shape, 1).to(cfg.device)

            y_Distr11 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)
            y_Distr12 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)

            y_Distr21 = torch.tensor([0 for i in range(batch_size)]).to(cfg.device)
            y_Distr22 = torch.tensor([1 for i in range(batch_size)]).to(cfg.device)

            # Classification losses
            loss_G1 = loss_fn_arr[0](out_G1, y_G1)
            loss_A2 = loss_fn_arr[0](out_A2, y_A2)
            loss_R3 = loss_fn_arr[0](out_R3, y_R3)
            loss_ID4 = loss_fn_arr[0](out_ID4, y_ID4)
            loss_Distr11 = loss_fn_arr[0](out_Distr1, y_Distr11)
            loss_Distr21 = loss_fn_arr[0](out_Distr2, y_Distr21)

            classification_loss = loss_G1 + loss_A2 + loss_R3 + loss_ID4 + loss_Distr11 + loss_Distr21
            test_loss += classification_loss.item()

            # Adversarial losses
            loss_A1 = loss_fn_arr[1](out_A1, y_A1)
            loss_R1 = loss_fn_arr[1](out_R1, y_R1)
            loss_ID1 = loss_fn_arr[1](out_ID1, y_ID1)
            
            loss_G2 = loss_fn_arr[2](out_G2, y_G2)
            loss_R2 = loss_fn_arr[2](out_R2, y_R2)
            loss_ID2 = loss_fn_arr[2](out_ID2, y_ID2)

            loss_G3 = loss_fn_arr[3](out_G3, y_G3)
            loss_A3 = loss_fn_arr[3](out_A3, y_A3)
            loss_ID3 = loss_fn_arr[3](out_ID3, y_ID3)

            loss_G4 = loss_fn_arr[4](out_G4, y_G4)
            loss_A4 = loss_fn_arr[4](out_A4, y_A4)
            loss_R4 = loss_fn_arr[4](out_R4, y_R4)

            loss_Distr12 = loss_fn_arr[5](out_Distr1, y_Distr12)
            loss_Distr22 = loss_fn_arr[5](out_Distr2, y_Distr22)

            adversarial_loss = loss_G2 + loss_G3 + loss_G4 + loss_A1 + loss_A3 + loss_A4 + loss_R1 + loss_R2 + loss_R4 + loss_ID1 + loss_ID2 + loss_ID3 + loss_Distr12 + loss_Distr22
            test_loss += adversarial_loss.item()

            # Calculate classifier accuracies
            correct_G += (out_G1.argmax(1) == y_G1).type(torch.float).sum().item()
            correct_A += (out_A2.argmax(1) == y_A2).type(torch.float).sum().item()
            correct_R += (out_R3.argmax(1) == y_R3).type(torch.float).sum().item()
            correct_ID += (out_ID4.argmax(1) == y_ID4).type(torch.float).sum().item()            
            correct_Distr += (out_Distr1.argmax(1) == y_Distr11).type(torch.float).sum().item()           
            correct_Distr += (out_Distr2.argmax(1) == y_Distr21).type(torch.float).sum().item()

    test_loss /= num_batches
    correct_G /= size
    correct_A /= size
    correct_R /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"Testing - Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_R: {(100*correct_R):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    test_loss_arr.append(test_loss)

def main(args):

    # get config
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    model = DebFace(cfg).to(cfg.device)
    # summary(model, (3, 112, 112))
    # print(cfg.load_weights)
    if cfg.load_weights:
        model.load_state_dict(torch.load(cfg.model_weights_dir + cfg.load_weights_file))

    weight_G = torch.tensor([(1/cfg.n_gender_classes) for i in range(cfg.n_gender_classes)]).to(cfg.device)
    weight_A = torch.tensor([(1/cfg.n_age_classes) for i in range(cfg.n_age_classes)]).to(cfg.device)
    weight_R = torch.tensor([(1/cfg.n_race_classes) for i in range(cfg.n_race_classes)]).to(cfg.device)
    weight_ID = torch.tensor([(1/cfg.n_id_classes) for i in range(cfg.n_id_classes)]).to(cfg.device)
    weight_Distr = torch.tensor([(1/cfg.n_distr_classes) for i in range(cfg.n_distr_classes)]).to(cfg.device)

    loss_fn_arr = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=weight_G), nn.CrossEntropyLoss(weight=weight_A), nn.CrossEntropyLoss(weight=weight_R), nn.CrossEntropyLoss(weight=weight_ID), nn.CrossEntropyLoss(weight=weight_Distr)]

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
    dataloader = []
    for i in range(2):
        X_tmp = torch.randn((10, 3, 112, 112))
        # y = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]])
        # assuming 4 classes each for gender, age, race and id
        y_tmp = torch.randint(2, (10, 4))
        dataloader.append((X_tmp, y_tmp))

    epochs = cfg.num_epoch

    try:
        os.makedirs(cfg.model_weights_dir)

    except:
        pass

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train(train_dataloader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg)
        train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, scheduler, cfg)
        # test(test_dataloader, model, loss_fn_arr, test_loss_arr, cfg)
        test(dataloader, model, loss_fn_arr, test_loss_arr, cfg)
        
        if cfg.save_model_weights_every > 0 and (t + 1)%cfg.save_model_weights_every == 0:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S_")
            torch.save(model.state_dict(), cfg.model_weights_dir + dt_string + f"debface_epoch_{t+1}_trial_" + cfg.trial_number + ".pth")

    if cfg.plot_losses:
        x = [i+1 for i in range(cfg.num_epoch)]
        plt.plot(x, train_loss_arr, 'g', label='train')
        plt.plot(x, test_loss_arr, 'r', label='test')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S_")
        plt.savefig(cfg.plots_dir + dt_string + "debface_trial_" + cfg.trial_number + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DebFace Training in Pytorch")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    main(parser.parse_args())
    print("DebFace Training completed successfully!")

    