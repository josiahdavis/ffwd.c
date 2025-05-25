import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

start = time.time()
torch.manual_seed(42)

batch_size = 128
hidden_size = 64
model_location = '/tmp/ffwd.bin'
data_location = '/tmp/data.bin'

# --------------
# SETUP
# --------------

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the model arch
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        bias = True
        self.linear_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear3 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_out = nn.Linear(hidden_size, 1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear_in(x))
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear_out(x)
        return x
    
# --------------
# DATA
# --------------

def read_data(data_dir = '/tmp/'):
    cols = pd.read_csv(os.path.join(data_dir, 'CaliforniaHousing/cal_housing.domain'), names=['col_names', 'col_type'], sep=":")
    data = pd.read_csv(os.path.join(data_dir, 'CaliforniaHousing/cal_housing.data'), header=None, names = cols.col_names)
    assert data.isnull().sum().sum() == 0, 'Missing Values in data'

    print(f'Read data into memory \n{data.head(8)} \n{data.describe()}')
    features, labels = data.iloc[:,:-1], data.iloc[:,-1]
    # Save for later convenience
    features.to_csv(os.path.join(data_dir, 'CaliforniaHousing/features.csv'), header=None, index=False)
    labels.to_csv(os.path.join(data_dir, 'CaliforniaHousing/labels.csv'), header=None, index=False)
    return features.values, labels.values

# Create dataset and dataloader
features, labels = read_data()
data_size, input_size = features.shape
print(f'Creating dataset with {data_size=}, {input_size=}')
dataset = MyDataset(features, labels)
test_len = int(0.1 * data_size)
train_len = data_size - test_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
print(f'{len(train_dataset)=:,} \n{len(test_dataset)=:,}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --------------
# TESTING 
# --------------

# Define model, loss function and optimizer
print(f'Creating model with {hidden_size=}')
model = RegressionModel(input_size, hidden_size)
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)


def eval(model, train_loader, test_loader, msg):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        train_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            train_loss += criterion(outputs, labels).item()
        train_loss /= len(train_loader)

        test_loss = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
        test_loss /= len(test_loader)
        print(f'{msg} | Train Loss = {train_loss:.4f} | Test Loss = {test_loss:.4f}.')

eval(model, train_loader, test_loader, "Before Training")

# --------------
# TRAINING 
# --------------

epochs = 100
for epoch in range(1, epochs + 1):
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

eval(model, train_loader, test_loader, "After Training")

# --------------
# SAVING
# --------------

# Save model along with a data sample
def write_tensor(tensor, handle):
    handle.write(tensor.detach().numpy().astype('float32').tobytes())

print(f'Saving model to {model_location} and data to {data_location}')
with open(model_location, 'wb') as file:
    write_tensor(model.linear_in.weight.t(), file)  # (C_in, C)
    write_tensor(model.linear_in.bias, file)        # (C,)

    write_tensor(model.linear1.weight.t(), file)    # (C, C)
    write_tensor(model.linear1.bias, file)          # (C,)

    write_tensor(model.linear2.weight.t(), file)    # (C, C)
    write_tensor(model.linear2.bias, file)          # (C,)

    write_tensor(model.linear3.weight.t(), file)    # (C, C)
    write_tensor(model.linear3.bias, file)          # (C,)

    write_tensor(model.linear_out.weight.t(), file) # (C, 1)
    write_tensor(model.linear_out.bias, file)       # (C,)

print(f'Saving data to {data_location}')
with open(data_location, 'wb') as data:
    # Data sample
    batch_features, batch_labels = next(iter(train_loader))
    write_tensor(batch_features, data)          # (B, C_in)
    write_tensor(batch_labels, data)            # (B, 1)

    # Expected output
    out_expected = model(batch_features)
    write_tensor(out_expected, data)   # (B, 1)

print(f'Predicting on all data \n{model(torch.tensor(features, dtype=torch.float32))[:8]}')
print(f'B={batch_size}, C_in={input_size}, C={hidden_size}')
print(f'W_out=\n{model.linear_out.weight.t()[:8, :8]}')
print(f'batch_labels ({batch_labels.shape})=\n{batch_labels[:8]}')
print(f'out_expected ({out_expected.shape})=\n{out_expected[:8]}')
print(f'Completed in {(time.time() - start) / 60:.2f} minutes')