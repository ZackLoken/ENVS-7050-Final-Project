# Import your Python modules
from ast import Num
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import classes, nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

##-------------------------------------------------------------------------------------------------##
##           Specify the folder containing images for inference and define the transforms          ##
##-------------------------------------------------------------------------------------------------##

data_dir = 'F:/Winter 21 Field Season/Training_data'

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])

##-------------------------------------------------------------------------------------------------##
##          Check for GPU availability, load your ResNet model, & set to evaluation mode           ##
##-------------------------------------------------------------------------------------------------##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('waterfowlMAV.pth')
model.eval()
# print(model)

##-------------------------------------------------------------------------------------------------##
##              Create a prediction function (requires Pillow image, not file path)                ##
##-------------------------------------------------------------------------------------------------##

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

##-------------------------------------------------------------------------------------------------##
##               Create a function that will pick random images from dataset folders               ##
##-------------------------------------------------------------------------------------------------##

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return loader, images, labels

##-------------------------------------------------------------------------------------------------##
##    Demo the prediction function; get random image sample, predict classes, & display results    ##
##-------------------------------------------------------------------------------------------------##

to_pil = transforms.ToPILImage()
loader, images, labels = get_random_images(64)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(loader.dataset.classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()

##-------------------------------------------------------------------------------------------------##
##                            Evaluate performance of the trained model                            ##
##-------------------------------------------------------------------------------------------------##

# Variables for computing model metrics
total = 0.0
correct = 0.0

# Evaluate the trained ResNet-50 model. In order to not have these operations
# tracked in the graphical outputs, wrap them in:
with torch.no_grad():
    for images, labels in loader:

        # Place features (images) and targets (labels) to GPU-
        images = images.to(device)
        labels = labels.to(device)
        # print(f"images.shape = {images.shape}, labels.shape = {labels.shape}")

        # Set model to evaluation mode-
        model.eval()

        # Make predictions using trained model-
        outputs = model(images)
        _, y_pred = torch.max(outputs, 1)
    
        # Total number of labels-
        total += labels.size(0)

        # Total number of correct predictions-
        correct += (y_pred == labels).sum()

    val_acc = 100 * (correct / total)
    print(f"ResNet-50 trained CNN's test metrics are:")
    print(f"accuracy = {val_acc:.2f}%, num of correct labels = {correct} & total num of labels = {total}")