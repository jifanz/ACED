import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
from sklearn import preprocessing
import numpy as np

dims = 512

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

crop_transform = transforms.Compose(
    [transforms.RandomCrop(6), transforms.ToTensor()])

trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=40)

testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=40)

patchset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=crop_transform)
patchloader = torch.utils.data.DataLoader(patchset, batch_size=len(trainset), shuffle=False, num_workers=40)

# Load data and filter the dataset as 2 vs 7
train_data, train_label = next(iter(trainloader))
mask = (train_label == 2) + (train_label == 7)
train_data, train_label = train_data[mask], train_label[mask] // 5

test_data, test_label = next(iter(testloader))
mask = (test_label == 2) + (test_label == 7)
test_data, test_label = test_data[mask], test_label[mask] // 5

# Using PCA to get features
train_data = train_data.view(train_data.size(0), -1)
test_data = test_data.view(test_data.size(0), -1)

U, S, V = torch.svd(train_data)
train_data = train_data @ V[:, :dims]
test_data = test_data @ V[:, :dims]

scaler = preprocessing.StandardScaler().fit(train_data.numpy())
train_data = scaler.transform(train_data.numpy())
test_data = scaler.transform(test_data.numpy())
train_label = train_label.numpy()
test_label = test_label.numpy()

print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

print(np.min(train_label), np.max(train_label))

with open("../data/svhn_{}.pkl".format(dims), 'wb') as file:
    pickle.dump([train_data.astype(float), train_label.astype(float), test_data.astype(float),
                 test_label.astype(float)], file, protocol=pickle.HIGHEST_PROTOCOL)
