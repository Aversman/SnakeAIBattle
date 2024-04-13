import numpy as np
from collections import deque
import torch
import random
from Model import QTrainer

MAX_MEMORY  = 100_000
BATCH_SIZE  = 1000
LR          = 1e-3

class Agent:
  def __init__(self, model):
    self.nGames = 0
    self.record = 0
    self.epsilon = 0.1 # randomness
    self.gamma = 0.9 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = model
    self.trainer = QTrainer(self.model, LR, self.gamma)

  def remember(self, state, action, reward, nextState, done):
    self.memory.append((state, action, reward, nextState, done))

  def trainLongMemory(self):
    if len(self.memory) > BATCH_SIZE:
      miniSample = random.sample(self.memory, BATCH_SIZE)
    else:
      miniSample = self.memory
    
    states, actions, rewards, nextStates, dones = zip(*miniSample)
    self.trainer.trainStep(states, actions, rewards, nextStates, dones)

  def trainShortMemory(self, state, action, reward, nextState, done):
    self.trainer.trainStep(state, action, reward, nextState, done)

  def getAction(self, state):
    finalMove = [0, 0, 0]
    #dangerousDirectionsSum = state[0] + state[1] + state[2]

    if self.nGames == 300:
      self.epsilon = 0

    """ if self.nGames == 100:
      self.epsilon = 0.3
    if self.nGames == 200:
      self.epsilon = 0.2
    if self.nGames == 300:
      self.epsilon = 0.1
    if self.nGames == 500:
      self.epsilon = 0 """

    if random.random() < self.epsilon:
      move = random.randint(0, 2)
      finalMove[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      finalMove[move] = 1
    
    return finalMove
