import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.__data = torch.zeros(1)
        self.__result = torch.zeros(1)

    def forward(self, data):
        output = self.linear(data)
        return output

    def train(self, data, result, epochs, lr):
        if not(torch.all(self.__data.eq(data)).item()) and not(torch.all(self.__data.eq(data)).item()):
            def check_dim(tensor):
                if tensor.dim() == 1:
                    tensor = tensor.reshape(-1, 1)
                    return tensor, 1
                else:
                    return tensor, tensor.shape[1]

            data, input_size = check_dim(data)
            result, output_size = check_dim(result)

            self.linear = nn.Linear(input_size, output_size)

            batch_size = 100
            train_dataset = TensorDataset(data, result)
            self.__train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            self.__data = data
            self.__result = result

        optimizer = torch.optim.SGD(self.linear.parameters(), lr)
        for epoch in range(epochs):
            for x, y in self.__train_loader:
                prediction = self(x)
                self.loss = nn.functional.mse_loss(prediction, y)
                self.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            yield self.loss.item()

    def predict(self, data):
        return(self(data))

    def reset(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

import pandas as pd
import seaborn as sns

df = pd.read_csv("./datasets_88705_204267_Real estate.csv")
df.drop(["No", "X1 transaction date"], axis=1, inplace=True)

data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
target = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

model = LinearModel()

# 100 training
epochs = 60
lr = 3e-8
loss_history = list(model.train(data, target, epochs=epochs, lr=lr))
print(model.loss)
plt.figure()
plt.plot(np.arange(len(loss_history)), loss_history, "b", lw=2)
plt.title("Epocs = {}, lr = {}".format(epochs, lr))

model.reset()
# 1000 training
epochs = 500
lr = 3e-8
loss_history = list(model.train(data, target, epochs=epochs, lr=lr))
print(model.loss)
plt.figure()
plt.plot(np.arange(len(loss_history)), loss_history, "b", lw=2)
plt.title("Epocs = {}, lr = {}".format(epochs, lr))

epochs = 500
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
