import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
from sklearn import preprocessing
import numpy as np


dims = 784

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

crop_transform = transforms.Compose(
    [transforms.RandomCrop(6), transforms.ToTensor()])

trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=40)

testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=40)

# Load data and filter the dataset as t-shirt vs pants.
train_data, train_label = next(iter(trainloader))
mask = (train_label == 0) + (train_label == 1)
train_data, train_label = train_data[mask], train_label[mask]

test_data, test_label = next(iter(testloader))
mask = (test_label == 0) + (test_label == 1)
test_data, test_label = test_data[mask], test_label[mask]

# Use original 784 features.
train_data = train_data.view(train_data.size(0), -1)
test_data = test_data.view(test_data.size(0), -1)

scaler = preprocessing.StandardScaler().fit(train_data.numpy())
train_data = scaler.transform(train_data.numpy())
test_data = scaler.transform(test_data.numpy())
train_label = train_label.numpy()
test_label = test_label.numpy()

print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

print(np.min(train_label), np.max(train_label))

with open("../data/fashion_{}.pkl".format(dims), 'wb') as file:
    pickle.dump([train_data.astype(float), train_label.astype(float), test_data.astype(float),
                 test_label.astype(float)], file, protocol=pickle.HIGHEST_PROTOCOL)
