import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from Unet import Unet

class isbi_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.image_files[idx].replace('volume', 'labels'))
        
        label = Image.open(label_name) 
        image = Image.open(img_name)

        if os.path.exists(label_name):
            label = Image.open(label_name)
        else:
            raise FileNotFoundError(f'Label file {label_name} not found.')
        
        if self.transform:
            image, label = self.transform(image, label)

        return image, label
    

class RandomTransforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()  
        ])
    
    def __call__(self, image, label):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        label = self.transform(label)
        return image, label

def train(model, dataloader, criterion, optimizer, scheduler, num_epochs=25):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
            for inputs, labels in t:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                t.set_postfix(loss=running_loss / ((t.n + 1) * dataloader.batch_size))
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        scheduler.step()

    print('Training complete')

model = Unet()  
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

transform = RandomTransforms()
dataset = isbi_Dataset(image_dir='/home/satvik/u-net/isbi-datasets/data/images', label_dir='/home/satvik/u-net/isbi-datasets/data/labels', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

train(model, dataloader, criterion, optimizer, scheduler)


# transform = RandomTransforms()
# dataset = SS_TEM_Dataset(image_dir='path/to/images', label_dir='path/to/labels', transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# model = Unet()
# model.to('cuda' if torch.cuda.is_available() else 'cpu')

# criterion = nn.BCEWithLogitsLoss() 
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  

# def train(model, dataloader, criterion, optimizer, scheduler, num_epochs=25):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
#             optimizer.zero_grad()
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
        
#         epoch_loss = running_loss / len(dataloader.dataset)
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
#         scheduler.step()
    
#     print('Training complete')


# train(model, dataloader, criterion, optimizer, scheduler)