# Import your Python modules
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

##-------------------------------------------------------------------------------------------------##
##    Define the train / validation dataset loader, using the SubsetRandomSampler for the split    ##
##-------------------------------------------------------------------------------------------------##

data_dir = 'F:/Winter 21 Field Season/Training_data'

def load_split_train_test(data_dir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),  # data augmentations are great
                                       transforms.RandomResizedCrop(224),  # for increasing the dataset size
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these
                                                            [0.229, 0.224, 0.225]) 
                                       ])

    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

##-------------------------------------------------------------------------------------------------##
##                 Check for GPU availability and load a pretrained ResNet50 model                 ##
##-------------------------------------------------------------------------------------------------##

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet50(pretrained=True) # Don't use pretrained for your actual master's work. 
# print(model)

##-------------------------------------------------------------------------------------------------##
##                Freeze pre-trained layers so you don't backpropagate through them                ##
##-------------------------------------------------------------------------------------------------##

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512), # Re-define the final fully-connected layer that will be trained with the images in the training dataset.
                                 nn.ReLU(),    # function which allows positive values to pass through, whereas negative values are modified to zero. 
                                 nn.Dropout(0.2), # if overfitting (training loss < validation loss) by a lot, try increasing your dropout value (e.g., 0.5)
                                 nn.Linear(512, len(trainloader.dataset.classes)),
                                 nn.LogSoftmax(dim=1)) # this is the output layer, it is a linear layer with LogSoftmax activiation
criterion = nn.NLLLoss() # create the loss function (criterion) -- NLLLoss is negative log-likelihood loss. NLLLoss & LogSoftmax act together as the cross-entropy loss. 
optimizer = optim.Adam(model.fc.parameters(), lr=0.003) # pick a model optimizer (Chose Adam because research says it's best & define learning rate--smaller lr for larger number of classes). Use stochastic gradient descent to determine weights. 
model.to(device) # send each parameter to the GPU one after another
# print(model)

##-------------------------------------------------------------------------------------------------##
##                                        Train the model                                          ##
##-------------------------------------------------------------------------------------------------##

epochs = 25 # ResNet model can be trained in 35 epochs. Start here, check validation loss, and continue increasing epochs until validation loss no longer improves. 
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

# This code chunk deals with displaying the losses and calculating accuracy every 3 batches. 
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))               
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'ResNet50_PyTorch.pth')

# During validation, be sure to set model.train() to model.eval().
# You can set back to model.train() after validation phase is completed. 

##-------------------------------------------------------------------------------------------------##
##                                     Plot the training loss                                      ##
##-------------------------------------------------------------------------------------------------##

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.title("ResNet 50 - Training Loss")
plt.xlabel("# of Epochal Steps (5 epochs/step)")
plt.ylabel("Loss")
plt.legend(frameon=False,
            loc = 'best')
plt.show()

##-------------------------------------------------------------------------------------------------##
##                                      Validate the model                                         ##
##-------------------------------------------------------------------------------------------------##

# epochs = 25 # ResNet model can be trained in 35 epochs. Start here, check validation loss, and continue increasing epochs until validation loss no longer improves. 
# steps = 0
# running_loss = 0
# print_every = 10
# train_losses, test_losses = [], []

# # This code chunk deals with displaying the losses and calculating accuracy every 3 batches. 
# for epoch in range(epochs):
#     for inputs, labels in trainloader:
#         steps += 1
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         logps = model.forward(inputs)
#         loss = criterion(logps, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#         if steps % print_every == 0:
#             test_loss = 0
#             accuracy = 0
#             model.eval()
#             with torch.no_grad():
#                 for inputs, labels in testloader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     logps = model.forward(inputs)
#                     batch_loss = criterion(logps, labels)
#                     test_loss += batch_loss.item()
                    
#                     ps = torch.exp(logps)
#                     top_p, top_class = ps.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

#             train_losses.append(running_loss/len(trainloader))
#             test_losses.append(test_loss/len(testloader))               
#             print(f"Epoch {epoch+1}/{epochs}.. "
#                   f"Train loss: {running_loss/print_every:.3f}.. "
#                   f"Test loss: {test_loss/len(testloader):.3f}.. "
#                   f"Test accuracy: {accuracy/len(testloader):.3f}")
#             running_loss = 0
#             model.eval()
# torch.save(model, 'ResNet50_PyTorch.pth')

##-------------------------------------------------------------------------------------------------##
##                            Plot the training and validation losses                              ##
##-------------------------------------------------------------------------------------------------##

# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Validation loss')
# plt.title("ResNet 50 - Validation Loss")
# plt.xlabel("# of Epochal Steps (5 epochs/step)")
# plt.ylabel("Loss")
# plt.legend(frameon=False,
#             loc = 'best')
# plt.show()

