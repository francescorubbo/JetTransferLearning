testdir = 'data/test'

import torch
model = torch.load('models/model.pth')
rndmodel = torch.load('models/model_rnd.pth')


import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Scale(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])

print('reading images')
test = datasets.ImageFolder(testdir, transform)

print('done reading. preparing loaders')

test_loader = torch.utils.data.DataLoader(
    test, batch_size=1, shuffle=False, num_workers=1)

import numpy as np
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()
preds = []
rnd_preds = []
ys = []
for inputs,labels in test_loader:
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(torch.Tensor(labels))
    outputs = model(inputs)
    toprob = torch.nn.Softmax()
    outputs = toprob(outputs)
    preds.append( outputs.data.cpu().numpy() )
    ys.append( labels.data.cpu().numpy()[0] )
    rnd_outputs = rndmodel(inputs)
    rnd_preds.append( rnd_outputs.data.cpu().numpy() )

import pandas as pd
testdf = pd.DataFrame(
    {'label': ys,
     'output': preds,
     'random': rnd_preds
    }
)
testdf.to_pickle('testdf.pkl')
