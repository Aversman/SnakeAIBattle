import numpy as np
from collections import deque
import torch
import random
from Model import QTrainer

MAX_MEMORY  = 100_000
BATCH_SIZE  = 1000
LR          = 1e-3

class Agent:
  def __init__(self, model, trainer):
    self.nGames = 0
    self.record = 0
    self.epsilon = 0 # randomness
    self.gamma = 0.9 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = model
    #self.trainer = QTrainer(self.model, LR, self.gamma)
    self.trainer =  trainer

  def remember(self, state, action, reward, nextState, isGameOver):
    self.memory.append((state, action, reward, nextState, isGameOver))

  def trainLongMemory(self):
    if len(self.memory) > BATCH_SIZE:
      miniSample = random.sample(self.memory, BATCH_SIZE)
    else:
      miniSample = self.memory
    
    states, actions, rewards, nextStates, isGameOvers = zip(*miniSample)
    self.trainer.trainStep(states, actions, rewards, nextStates, isGameOvers)

  def trainShortMemory(self, state, action, reward, nextState, isGameOver):
    self.trainer.trainStep(state, action, reward, nextState, isGameOver)

  def getAction(self, state):
    self.epsilon = 80 - self.nGames
    finalMove = [0, 0, 0]
    
    if random.randint(0, 200) < self.epsilon:
      move = random.randint(0, 2)
      finalMove[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      finalMove[move] = 1
    
    return finalMove
