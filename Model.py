import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Model 1
class LinearQNetModel1(nn.Module):
  def __init__(self, inputSize, hiddenSize, outputSize):
    super().__init__()
    self.linear1 = nn.Linear(inputSize, hiddenSize)
    self.linear2 = nn.Linear(hiddenSize, int(hiddenSize / 2))
    self.linear3 = nn.Linear(int(hiddenSize / 2), int(hiddenSize / 4))
    self.linear4 = nn.Linear(int(hiddenSize / 4), outputSize)
  
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x
  
  def save(self, filename='model1.pth'):
    modelFolderPath = 'models'
    
    if not os.path.exists(modelFolderPath):
      os.makedirs(modelFolderPath)
    
    filename = os.path.join(modelFolderPath, filename)
    torch.save(self.state_dict(), filename)

# Model 2
class LinearQNetModel2(nn.Module):
  def __init__(self, inputSize, hiddenSize, outputSize):
    super().__init__()
    self.linear1 = nn.Linear(inputSize, hiddenSize)
    self.linear2 = nn.Linear(hiddenSize, int(hiddenSize / 2))
    self.linear3 = nn.Linear(int(hiddenSize / 2), int(hiddenSize / 4))
    self.linear4 = nn.Linear(int(hiddenSize / 4), outputSize)
  
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    x = self.linear4(x)
    return x
  
  def save(self, filename='model2.pth'):
    modelFolderPath = 'model'
    
    if not os.path.exists(modelFolderPath):
      os.makedirs(modelFolderPath)
    
    filename = os.path.join(modelFolderPath, filename)
    torch.save(self.state_dict(), filename)


class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()
  
  def trainStep(self, state, action, reward, nextState, done):
    state       = torch.tensor(state, dtype=torch.float)
    nextState   = torch.tensor(nextState, dtype=torch.float)
    action      = torch.tensor(action, dtype=torch.long)
    reward      = torch.tensor(reward, dtype=torch.float)

    if len(state.shape) == 1:
      state       = torch.unsqueeze(state, 0)
      nextState   = torch.unsqueeze(nextState, 0)
      action      = torch.unsqueeze(action, 0)
      reward      = torch.unsqueeze(reward, 0)
      done        = (done, )

    pred = self.model(state)
    target = pred.clone()

    for idx in range(len(done)):
      Q_new = reward[idx]
      if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))
      
      target[idx][torch.argmax(action[idx]).item()] = Q_new

    # 2: Q_new = r + y * max(next_predicted Q value)
    
    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()
    self.optimizer.step()
