import torch
from dataset import PeptidesDataset
from model import NeuralNetwork
from torch.utils.data import DataLoader, random_split
from torch.nn import Module
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# CUDA for Pytorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 20
DEBUG = True


def train(dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer,
          ) -> None:
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = torch.squeeze(model(X))
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader: DataLoader,
         model: Module,
         loss_fn: Module
         ) -> Tuple[float, float]:
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = torch.squeeze(model(X))
            test_loss += loss_fn(pred, y).item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_test_loss = test_loss / len(dataloader)
    accuracy = correct / total

    return avg_test_loss, accuracy


def epoch(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer
          ) -> Tuple[float, float, float, float]:
    train(train_dataloader, model, loss_fn, optimizer)
    train_loss, train_accuracy = test(train_dataloader, model, loss_fn)
    test_loss, test_accuracy = test(test_dataloader, model, loss_fn)
    return train_loss, train_accuracy, test_loss, test_accuracy

def plot_train_test_loss(train_losses: List,
                         test_losses: List,
                         path: str = None):
    plt.plot(test_losses, 'r', label='test loss')
    plt.plot(train_losses, 'b', label='train loss')
    plt.xlabel('Epochs (#)')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend(loc="upper right")
    plt.title('Train/Test Loss Over Epochs')
    
    plt.show()
    if path:
        plt.savefig(path)

    

def main():

    # Creating data set from given files
    dataset = PeptidesDataset('../dataset/pos_A0201.txt', '../dataset/neg_A0201.txt')

    # Define size of train and test datasets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    #
    model = NeuralNetwork()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []
    epochs = range(NUMBER_OF_EPOCHS)
    for epoch_index in epochs:
        train_loss, train_accuracy, test_loss, test_accuracy = epoch(
            train_dataloader, test_dataloader, model, loss_fn, optimizer
            )
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if DEBUG:
            print('-------------------')
            print(f'Epoch {epoch_index} is complete.')
            print('\tTrain\tTest')
            print(f'Loss\t{train_loss}\t{test_loss}')
            print(f'Accuracy\t{train_accuracy}\t{test_accuracy}')
            print('-------------------')

    plot_train_test_loss(train_losses, test_losses)

if __name__ == '__main__':
    main()
