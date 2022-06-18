import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchsummary import summary
# from torchviz import make_dot

from utils.utils_config import get_config

from backbones.debface import DebFace
# from backbones.am_softmax import Am_softmax
from utils.utils_config import ConfigParams

def train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, cfg):
    # size = len(dataloader.dataset)
    size = 100 # size of dataset
    num_batches = len(dataloader)
    batch_size = int(size/num_batches)
    
    model.train()
    train_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(cfg.device)
        X.requires_grad = True

        out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2 = model(X)

        y_G1 = y[:, 0].clone()
        y_A1 = torch.full(y_G1.shape, 1)
        y_R1 = torch.full(y_G1.shape, 1)
        y_ID1 = torch.full(y_G1.shape, 1)

        y_A2  = y[:, 1].clone()
        y_G2  = torch.full(y_A2.shape, 1)
        y_R2  = torch.full(y_A2.shape, 1)
        y_ID2 = torch.full(y_A2.shape, 1)

        y_R3  = y[:, 2].clone()
        y_G3  = torch.full(y_R3.shape, 1)
        y_A3  = torch.full(y_R3.shape, 1)
        y_ID3 = torch.full(y_R3.shape, 1)

        y_ID4  = y[:, 3].clone()
        y_G4  = torch.full(y_ID4.shape, 1)
        y_A4  = torch.full(y_ID4.shape, 1)
        y_R4 = torch.full(y_ID4.shape, 1)

        y_Distr11 = torch.tensor([1 for i in range(batch_size)])
        y_Distr12 = torch.tensor([1 for i in range(batch_size)])

        y_Distr21 = torch.tensor([0 for i in range(batch_size)])
        y_Distr22 = torch.tensor([1 for i in range(batch_size)])

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
            loss, current = (classification_loss.item() + adversarial_loss.item()), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
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

    train_loss /= num_batches
    correct_G /= size
    correct_A /= size
    correct_R /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"\nTraining - Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_R: {(100*correct_R):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    # # torch.save(model.state_dict(), cfg.path +"run1.pth")

    train_loss_arr.append(train_loss)

def test(dataloader, model, loss_fn_arr, test_loss_arr, cfg):
    # size = len(dataloader.dataset)
    size = 100 # size of dataset
    num_batches = len(dataloader)
    batch_size = int(size/num_batches)
    test_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(cfg.device)
            X.requires_grad = True

            out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2 = model(X)

            y_G1 = y[:, 0].clone()
            y_A1 = torch.full(y_G1.shape, 1)
            y_R1 = torch.full(y_G1.shape, 1)
            y_ID1 = torch.full(y_G1.shape, 1)

            y_A2  = y[:, 1].clone()
            y_G2  = torch.full(y_A2.shape, 1)
            y_R2  = torch.full(y_A2.shape, 1)
            y_ID2 = torch.full(y_A2.shape, 1)

            y_R3  = y[:, 2].clone()
            y_G3  = torch.full(y_R3.shape, 1)
            y_A3  = torch.full(y_R3.shape, 1)
            y_ID3 = torch.full(y_R3.shape, 1)

            y_ID4  = y[:, 3].clone()
            y_G4  = torch.full(y_ID4.shape, 1)
            y_A4  = torch.full(y_ID4.shape, 1)
            y_R4 = torch.full(y_ID4.shape, 1)

            y_Distr11 = torch.tensor([1 for i in range(batch_size)])
            y_Distr12 = torch.tensor([1 for i in range(batch_size)])

            y_Distr21 = torch.tensor([0 for i in range(batch_size)])
            y_Distr22 = torch.tensor([1 for i in range(batch_size)])

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

    weight_G = torch.tensor([(1/cfg.n_gender_classes) for i in range(cfg.n_gender_classes)])
    weight_A = torch.tensor([(1/cfg.n_age_classes) for i in range(cfg.n_age_classes)])
    weight_R = torch.tensor([(1/cfg.n_race_classes) for i in range(cfg.n_race_classes)])
    weight_ID = torch.tensor([(1/cfg.n_id_classes) for i in range(cfg.n_id_classes)])
    weight_Distr = torch.tensor([(1/cfg.n_distr_classes) for i in range(cfg.n_distr_classes)])

    loss_fn_arr = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=weight_G), nn.CrossEntropyLoss(weight=weight_A), nn.CrossEntropyLoss(weight=weight_R), nn.CrossEntropyLoss(weight=weight_ID), nn.CrossEntropyLoss(weight=weight_Distr)]

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loss_arr = []
    test_loss_arr = []

    # creating a random dataset (same shape as the facial dataset we will be using) for testing the code logic
    dataloader = []
    for i in range(10):
        X_tmp = torch.randn((10, 3, 112, 112))
        # y = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]])
        # assuming 4 classes each for gender, age, race and id
        y_tmp = torch.randint(2, (10, 4))
        dataloader.append((X_tmp, y_tmp))

    epochs = cfg.num_epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train(train_dataloader, model, loss_fn_arr, train_loss_arr, optimizer, cfg)
        train(dataloader, model, loss_fn_arr, train_loss_arr, optimizer, cfg)
        # test(test_dataloader, model, loss_fn_arr, test_loss_arr, cfg)
        test(dataloader, model, loss_fn_arr, test_loss_arr, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DebFace Training in Pytorch")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    main(parser.parse_args())
    print("DebFace Training completed successfully!")

    