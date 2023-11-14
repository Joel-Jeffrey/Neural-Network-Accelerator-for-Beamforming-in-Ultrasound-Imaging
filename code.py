!pip install monai

# Cell 1: Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import libraries for the Neural Network Section
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f # Activation function (ReLU)
import torch.utils.data as DataLoader
import torch.nn.init as init

from google.colab import drive
from scipy.signal import hilbert
from monai.networks.layers import HilbertTransform

# Cell 2: Setting variables

start_epoch = 0
epochs = 5
stop_epoch = start_epoch + epochs

train_scans = 10  # Number of scanning data frames

# Image/Time of Flight data dimensions. Modify according to need
rows = 374
cols = 128
channels = 128
batch_size = 5

'''
# It is preferable to name the scan data as follows:
# Time of flight data: Dimension: Rows x Cols x Channels
# MVDR Data prior to hilbert transform & log compression: Dimension: Rows x Cols

# Name them with numbers as indexes. An example is. tofc_1, tofc_2 and mvdr_1, mvdr_2
# Naming them with numerical index gives an advantage to pick them for training.
'''

# Cell 3: Making the network

TOFC_Dataset = input("Enter path to TOFC dataset: (Eg - '/content/gdrive/MyDrive/Dataset/tofc/x_')\n")
MVDR_Dataset = input("\nEnter path to MVDR dataset: (Eg - '/content/gdrive/MyDrive/Dataset/mvdr/y_')\n")
Model_Path = input("\nEnter path to store model: (Eg - '/content/gdrive/My Drive/Dataset/model.pth')\n")
class Antirectifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x-torch.mean(x , dim=1 , keepdim=True)
        x = f.normalize(x , p=2 , dim = 1)
        pos = f.relu(x)
        neg = f.relu(-x)
        a = torch.cat([pos , neg] , 1)
        return a

# CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Make sure to call the superclass constructor properly
        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = 32, kernel_size=3, padding=1)  # Set padding to 1 for 'same' padding
        self.antirect1 = Antirectifier()
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=3, padding=1)  # Set padding to 1 for 'same' padding
        self.antirect2 = Antirectifier()
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1)  # Set padding to 1 for 'same' padding
        self.antirect3 = Antirectifier()
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=3, padding=1)  # Set padding to 1 for 'same' padding
        self.antirect4 = Antirectifier()
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = channels, kernel_size=3, padding=1)  # Set padding to 1 for 'same' padding

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x = x / x_norm
        x = self.conv1(x)
        x = self.antirect1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.antirect2(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.antirect3(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        x = self.antirect4(x)
        x = self.batchnorm4(x)
        x = self.conv5(x)
        x = f.softmax(x,dim=2)
        return x

model = Net()
print(model)

# Cell 4: Compiling the model

def msle(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    first_log = torch.log(torch.clamp(torch.abs(y_pred), torch.finfo(y_pred.dtype).eps))
    second_log = torch.log(torch.clamp(torch.abs(y_true), torch.finfo(y_true.dtype).eps))
    return torch.mean(torch.square(first_log - second_log))

def mse(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    return torch.mean(torch.square(y_true - y_pred))

learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = mse

# Cell 5: Train the model

# Loading Dataset and Training
drive.mount('/content/gdrive')

'''
# Best to save data with the number 1 to 600 as it will be easy to pick them
# to train as mentioned above in cell 2.
'''

# Produces indexed for picking data to train
list_train = np.add(1, np.arange(train_scans))

X_final = torch.zeros(batch_size, channels, rows, cols)
# X_final = torch.zeros(batch_size, rows,cols,channels)
y_final = torch.zeros(batch_size, rows,cols)
# y_final = torch.zeros(batch_size, rows, cols)
a=0

for epoch in range(start_epoch, stop_epoch):  # Run through all epochs
    np.random.shuffle(list_train)  # Randomize the entry of data to train
    for batch in range(0, train_scans, batch_size):
        # To load one batch at a time to prevent memory issues
        for scan in range(batch_size):
            j = list_train[batch + scan]
            # filename_x = '/content/gdrive/MyDrive/Dataset/tofc/x_'+str(j)+'.csv'
            # filename_y = '/content/gdrive/MyDrive/Dataset/mvdr/y_'+str(j)+'.csv'
            filename_x = TOFC_Dataset+str(j)+'.csv'
            filename_y = MVDR_Dataset+str(j)+'.csv'
            x_data = pd.read_csv(filename_x, header = None)
            y_data = pd.read_csv(filename_y, header = None)

            X= x_data.iloc[:,:].values
            X = np.reshape(X,(channels, rows, cols),order ='F')
            # X = np.reshape(X,(rows,cols,channels),order ='F')
            y= y_data.iloc[:].values
            y= np.reshape(y,(rows,cols), order='F')
            # y= np.reshape(y,(rows,cols), order='F')

            # Load your data here one by one by using j as index.
            # As mentioned in cell 2, e.g. 'tofc_'+str(j) or 'mvdr_'+str(j)

            # X = load_ToFC_data_using_j_as_index  # dimensions:(rows, cols, channels)
            # y = load_mvdr_data_using_j_as_index  # dimensions:(rows, cols)

            X_final[scan, :, :, :] = torch.from_numpy(X)  # storing loaded data as a batch
            y_final[scan, :, :] = torch.from_numpy(y)  # storing loaded data as a batch

        X_final, y_final = Variable(X_final), Variable(y_final)
        optimizer.zero_grad()
        outputs = model(X_final)
        beamformed = torch.mul(outputs , X_final)
        beamformed_sum = torch.sum(beamformed, 1)
        beamformed_sum = beamformed_sum.permute([0,2,1])
        for i in range(beamformed_sum.size(0)):
          for j in range(128):
            h_transform = HilbertTransform(axis=0)
            beamformed_sum[i][j] = h_transform(beamformed_sum[i][j])
            # beamformed_sum[i][j] = beamformed_sum[i][j].detach().numpy()
            # beamformed_sum[i][j] = hilbert(beamformed_sum[i][j],axis=0)
            # X = layer(axis=0,n=beamformed_sum.shape[0])
            # beamformed_sum[i][j] = X.forward(beamformed_sum)
        envelope = torch.abs(beamformed_sum)
        envelope = envelope.permute([0 ,2,1])
        outputs = 20*torch.log10(envelope/torch.max(envelope))
        loss = mse(y_final,outputs)
        loss.backward(retain_graph=True)
        optimizer.step()
        if(a==0):
          a=1
          fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))
          ax1.set_title("CNN")
          im1 = ax1.imshow(outputs[0,:,:].detach().numpy(), cmap='gray')
          divider = make_axes_locatable(ax1)
          cax1 = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(im1, cax = cax1)

          ax2.set_title("MVDR")
          im2 = ax2.imshow(y_final, cmap='gray')
          divider = make_axes_locatable(ax2)
          cax2 = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(im2, cax= cax2)


    # file_name_w = '/content/gdrive/My Drive/Dataset/model.pth'  # Enter your path
    file_name_w = Model_Path
    torch.save(model.state_dict(), file_name_w)




