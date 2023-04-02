import time
import numpy as np
import math
import torch
from torch import nn
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
 
device = "cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.activation = nn.ReLU()
    self.pool = nn.MaxPool2d(2, stride = 2)
    self.convlayers = nn.ModuleList([
        # shape = 1 * 28 * 28
        nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5),
        # shape = 3 * 24 * 24 -> pool -> shape = 3 * 12 * 12
        nn.Conv2d(in_channels = 3, out_channels = 9, kernel_size = 5)
        # shape = 3 * 8 * 8 -> pool -> shape = 9 * 4 * 4
    ])
    self.linearlayers = nn.ModuleList([
        # shape = (9 * 4 * 4)
        nn.Linear(9 * 4 * 4, 100),
        nn.Linear(100, 100),
        nn.Linear(100, 10)
    ])
  @autocast()
  def forward(self, x):
    for i, conv in enumerate(self.convlayers):
        x = conv(x) 
        x = self.activation(x)
        x = self.pool(x)
    x = torch.flatten(x, 1)
    for i, linear in enumerate(self.linearlayers): 
        x = linear(x)
        if i != len(self.linearlayers) - 1: 
            x = self.activation(x)
    return x

learning_rate = 1e-3
epochs = 10
batch_size = 100
input_size = 28 * 28
output_size = 10

model = Net()
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = torchvision.datasets.MNIST(root = "./data", train = False, transform = transform, download = True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


datasize = len(train_data)
n_iterations = math.ceil(datasize / batch_size)

datapoints = [[], []]
for epoch in range(epochs):
  avg_loss = 0
  for i, [images, labels] in enumerate(train_dataloader):
    optimizer.zero_grad()
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    avg_loss += loss.item() * images.shape[0]
    if (i + 1) % 100 == 0:
      print(f"Loss: {loss:.7f} [iteration {i + 1}/{n_iterations} in epoch {epoch + 1}/{epochs}]")
  avg_loss /= datasize   

  datapoints[0].append(epoch + 1)
  datapoints[1].append(avg_loss)
  if (epoch + 1) % 10 == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }
    filename = "./checkpoints/" + time.asctime() + ".pt"
    torch.save(checkpoint, filename)

import matplotlib.pyplot as plt
plt.style.use('classic')
fig = plt.figure()
loss_fig = fig.add_subplot(1,1,1)
loss_fig.set_title("Loss curve")
loss_fig.set_xlabel("Epochs")
loss_fig.set_ylabel("Loss")
loss_fig.plot(datapoints[0], datapoints[1])
plt.show()

correct = 0
with torch.no_grad():
  for i, [images, labels] in enumerate(test_dataloader):
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)
    _, guess = torch.max(pred, dim = 1)
    correct += torch.sum(guess == labels)
print(f"Accracy: {correct / len(test_data) * 100}% in {len(test_data)} tests")
