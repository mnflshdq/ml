import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        output = self.linear(data)
        return output

    def train(self, data, result, epochs, lr):
        def check_dim(tensor):
            if tensor.dim() == 1:
                tensor.reshape(-1, 1)
                return 1
            else:
                return tensor.shape[1]

        self.linear = nn.Linear(check_dim(data), check_dim(result))

        batch_size = 100
        train_dataset = TensorDataset(data, result)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        optimizer = torch.optim.SGD(self.linear.parameters(), lr)
        for epoch in range(epochs):
            for x, y in train_loader:
                prediction = self(x)
                self.loss = nn.functional.mse_loss(prediction, y)
                self.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            yield self.loss.item()

    def predict(self, data):
        return(self(data))

    def reset_param(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

import pandas as pd
import seaborn as sns

df = pd.read_csv("./datasets_88705_204267_Real estate.csv")
df.drop(["No", "X1 transaction date"], axis=1, inplace=True)

data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
target = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
target = target.reshape(-1, 1)

model = LinearModel()

# 100 training
epochs = 100
lr = 3e-8
loss_history = list(model.train(data, target, epochs=epochs, lr=lr))
print(model.loss)
plt.figure()
plt.plot(np.arange(len(loss_history)), loss_history, "b", lw=2)
plt.title("Epocs = {}, lr = {}".format(epochs, lr))

model.reset_param()
# 1000 training
epochs = 1000
lr = 3e-8
loss_history = list(model.train(data, target, epochs=epochs, lr=lr))
print(model.loss)
plt.figure()
plt.plot(np.arange(len(loss_history)), loss_history, "b", lw=2)
plt.title("Epocs = {}, lr = {}".format(epochs, lr))

# Compare
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(data, target)
Yhat = lr.predict(data)
loss = mean_squared_error(Yhat, target)
print(loss)

plt.show()
