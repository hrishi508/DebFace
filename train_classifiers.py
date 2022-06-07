import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.utils_config import get_config

from backbones.debface import DebFace
from backbones.am_softmax import Am_softmax
from utils.utils_config import ConfigParams

def train(dataloader, model, loss_fn_array, train_loss_arr, optimizer, cfg):
    size = len(dataloader.dataset)
    
    model.train()
    train_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(cfg.device)
        X.requires_grad = True

        out_G, out_A, out_R, out_ID, out_Distr1, out_Distr2 = model(X)

        loss_G = loss_fn_array[0](out_G, y_G)
        loss_A = loss_fn_array[1](out_A, y_A)
        loss_R = loss_fn_array[1](out_R, y_R)
        loss_ID = loss_fn_array[2](out_ID, y_ID)
        loss_Distr1 = loss_fn_array[0](out_Distr1, y_Distr1)
        loss_Distr2 = loss_fn_array[0](out_Distr2, y_Distr2)

        loss = loss_G + loss_A + loss_R + loss_ID + loss_Distr1 + loss_Distr2
        train_loss += loss.item()

        with torch.no_grad():
            correct_G += (out_G.argmax(1) == y_G).type(torch.float).sum().item()
            correct_A += (out_A.argmax(1) == y_A).type(torch.float).sum().item()
            correct_R += (out_R.argmax(1) == y_R).type(torch.float).sum().item()
            correct_ID += (out_ID.argmax(1) == y_ID).type(torch.float).sum().item()
            correct_Distr += (out_Distr1.argmax(1) == y_Distr1).type(torch.float).sum().item()
            correct_Distr += (out_Distr2.argmax(1) == y_Distr2).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch!=0 and batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= len(dataloader)
    correct_G /= size
    correct_A /= size
    correct_R /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_R: {(100*correct_R):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    torch.save(model.state_dict(), cfg.path +"run1.pth")

    train_loss_arr.append(train_loss)

def test(dataloader, model, loss_fn_array, test_loss_arr):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct_G, correct_A, correct_R, correct_ID, correct_Distr = 0, 0, 0, 0, 0 

    with torch.no_grad():
        for X, y in dataloader:
            out_G, out_A, out_R, out_ID, out_Distr1, out_Distr2 = model(X)

            loss_G = loss_fn_array[0](out_G, y_G)
            loss_A = loss_fn_array[1](out_A, y_A)
            loss_R = loss_fn_array[1](out_R, y_R)
            loss_ID = loss_fn_array[2](out_ID, y_ID)
            loss_Distr1 = loss_fn_array[0](out_Distr1, y_Distr1)
            loss_Distr2 = loss_fn_array[0](out_Distr2, y_Distr2)
            
            loss = loss_G + loss_A + loss_R + loss_ID + loss_Distr1 + loss_Distr2
            test_loss += loss.item()

            correct_G += (out_G.argmax(1) == y_G).type(torch.float).sum().item()
            correct_A += (out_A.argmax(1) == y_A).type(torch.float).sum().item()
            correct_R += (out_R.argmax(1) == y_R).type(torch.float).sum().item()
            correct_ID += (out_ID.argmax(1) == y_ID).type(torch.float).sum().item()
            correct_Distr += (out_Distr1.argmax(1) == y_Distr1).type(torch.float).sum().item()
            correct_Distr += (out_Distr2.argmax(1) == y_Distr2).type(torch.float).sum().item()

    test_loss /= num_batches
    correct_G /= size
    correct_A /= size
    correct_R /= size
    correct_ID /= size
    correct_Distr /= (size * 2)
    print(f"Accuracy_G: {(100*correct_G):>0.1f}%, Accuracy_A: {(100*correct_A):>0.1f}%, Accuracy_R: {(100*correct_R):>0.1f}%, Accuracy_ID: {(100*correct_ID):>0.1f}%, Accuracy_Distr: {(100*correct_Distr):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    test_loss_arr.append(test_loss)

def main(args):

    # get config
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    model = DebFace(cfg).to(cfg.device)
    summary(model, (3, 112, 112))

    loss_fn_array = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), Am_softmax(cfg.embedding_size, cfg.n_id_classes)]
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    train_loss_arr = []
    test_loss_arr = []

    epochs = cfg.num_epoch
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn_array, train_loss_arr, optimizer, cfg)
    #     test(test_dataloader, model, loss_fn_array, test_loss_arr)
    # print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DebFace Training in Pytorch")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    main(parser.parse_args())
    print("DebFace Training completed successfully!")

    