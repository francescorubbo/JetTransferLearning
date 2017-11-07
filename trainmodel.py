import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    use_gpu = torch.cuda.is_available()
    if use_gpu: print('Using GPU')
    else: print('No GPU!!')
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    for epoch in range(num_epochs):
        print('Epoch',epoch)
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]
                
            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc))
            
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def getmodel(num_classes,model,train_dl,valid_dl,num_epochs=10):
    
    use_gpu = torch.cuda.is_available()
    inputs, labels = next(iter(train_dl))
    if use_gpu:
        model = model.cuda()
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    print( outputs.size() )

    # freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # new final layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    if use_gpu:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    dloaders = {'train':train_dl, 'valid':valid_dl}
    
    model = train_model(dloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

    return model
