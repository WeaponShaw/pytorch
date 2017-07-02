#!/usr/bin/env ipython

import file_name_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def mnist_decoder(filename):
    return int(filename.split('.')[0].split('_')[-1])


def train():
    train_dataset = file_name_dataset.FileNameDataset('./images/train', mnist_decoder)
    train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('%d batch, loss = %f' % (batch_idx, loss.data[0]))
    return model


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    test_dataset = file_name_dataset.FileNameDataset('./images/test', mnist_decoder)
    test_loader = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size = 32, shuffle = True, num_workers = 0)
    for data, target in test_loader:
        data, target = Variable(data, volatile = True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    print('acc = %f' % (100. * correct / len(test_dataset)))


if __name__ == '__main__':
    model = train()
    test(model)
