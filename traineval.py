import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import KingsVisionNN2
from preprocess import fen_to_vector, ChessDataset, eval_to_int
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def mse_loss(predictions, targets):
    loss = torch.mean((predictions - targets) ** 2)
    return loss
####### train eval
def AdamW_main():
    MAX_DATA = 10000  # 10k max
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    print("Preparing Data...")
    data = pd.read_csv("datasets/chess evaluations/chessData.csv")
    data = data[:MAX_DATA]
    data["Evaluation"] = data["Evaluation"].map(eval_to_int)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    trainset = ChessDataset(train_data)
    testset = ChessDataset(test_data)

    batch_size = 10

    print("Converting to PyTorch Dataset...")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = KingsVisionNN2().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters())

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (200 * batch_size)))
                running_loss = 0.0

    print('Finished Training')

    PATH = './chess.pth'
    torch.save(net.state_dict(), PATH)

    print('Evaluating model')

    count = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            count += len(labels)
            total_loss += loss.item()
            if count % 1000 == 0:
                print('Average error of the model on the {} tactics positions is {}'.format(count, loss/count))
                
                
if __name__ == "__main__":
    AdamW_main()
