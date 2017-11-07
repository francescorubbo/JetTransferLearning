traindir = 'data/train'
valdir = 'data/val'

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Scale(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])

print('reading images')

train = datasets.ImageFolder(traindir, transform)
valid = datasets.ImageFolder(valdir, transform)

print('done reading. preparing loaders')

train_loader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(
    valid, batch_size=32, shuffle=True, num_workers=4)

print('setting up model and training')

import torchvision.models as models
from trainmodel import getmodel
resnet152 = models.resnet18(pretrained=True)
model = getmodel(2,resnet152,train_loader,valid_loader)

torch.save(model,'models/model.pth')
