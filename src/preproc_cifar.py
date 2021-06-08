import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
from sklearn import preprocessing
import numpy as np


num_patches = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

crop_transform = transforms.Compose(
    [transforms.RandomCrop(6), transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=40)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=40)

patchset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=crop_transform)
patchloader = torch.utils.data.DataLoader(patchset, batch_size=len(trainset), shuffle=False, num_workers=40)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load data and filter the dataset as plane vs bird
train_data, train_label = next(iter(trainloader))
mask = (train_label == 0) + (train_label == 2)
train_data, train_label = train_data[mask], train_label[mask] // 2

test_data, test_label = next(iter(testloader))
mask = (test_label == 0) + (test_label == 2)
test_data, test_label = test_data[mask], test_label[mask] // 2

# Load random patches
patches, _ = next(iter(patchloader))
rnd = np.random.RandomState(12345)
patches = patches[rnd.choice(np.arange(0, patches.size(0)), num_patches, replace=False)]

# # Convolve and pool to get features
# train_data = F.conv2d(train_data, patches)
# train_data = F.avg_pool2d(train_data, 15, stride=6).view(train_data.size(0), -1)
#
# test_data = F.conv2d(test_data, patches)
# test_data = F.avg_pool2d(test_data, 15, stride=6).view(test_data.size(0), -1)
#

# Using PCA to get features
train_data = train_data.view(train_data.size(0), -1)
test_data = test_data.view(test_data.size(0), -1)

U, S, V = torch.svd(train_data)
train_data = train_data @ V[:, :num_patches * 9]
test_data = test_data @ V[:, :num_patches * 9]

scaler = preprocessing.StandardScaler().fit(train_data.numpy())
train_data = scaler.transform(train_data.numpy())
test_data = scaler.transform(test_data.numpy())
train_label = train_label.numpy()
test_label = test_label.numpy()

print(train_data.shape, train_label.shape, test_data.shape, test_label.shape, patches.size(), type(patches))

print(np.min(train_label), np.max(train_label))

with open("../data/cifar_{}.pkl".format(num_patches), 'wb') as file:
    pickle.dump([train_data.astype(float), train_label.astype(float), test_data.astype(float),
                 test_label.astype(float)], file, protocol=pickle.HIGHEST_PROTOCOL)
