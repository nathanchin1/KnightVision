import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import KnightVisionNN
from preprocess import fen_to_vector, ChessDataset, eval_to_int, encode_moves, TopMovesDataset
from sklearn.model_selection import train_test_split
#from sklearn.metrics import precision_score, recall_score, f1_score, top_k_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data5000 = pd.read_csv("datasets/top_move5000.csv")
def AdamW_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    print("Preparing Data...")
    data = data5000
    all_moves = set()
    for top_moves in data5000['top_moves']:
        moves = eval(top_moves)  
        all_moves.update(moves)
    move_directory = {move: idx for idx, move in enumerate(all_moves)}
    data["EncodedMoves"] = data["top_moves"].apply(lambda moves: encode_moves(eval(moves), move_directory))
    
    # use dataloader to convert to pytorch tensors
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    trainset = TopMovesDataset(train_data, move_directory)
    testset = TopMovesDataset(test_data, move_directory)

    
    batch_size = 10
    print("Converting to PyTorch Dataset...")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_moves = len(move_directory)
    net = KnightVisionNN(num_moves).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss
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
            if count % 500 == 0:
                print('Average error of the model on the {} tactics positions is {}'.format(count, total_loss / count))
    
if __name__ == "__main__":
    AdamW_main()


