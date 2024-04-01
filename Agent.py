import numpy as np
from collections import deque
import torch
import random
from Model import QTrainer

MAX_MEMORY  = 100000
BATCH_SIZE  = 1000
LR          = 4e-4

class Agent:
  def __init__(self, model):
    self.nGames = 0
    self.record = 0
    self.epsilon = 0.3 # randomness
    self.gamma = 0.9 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = model
    self.trainer = QTrainer(self.model, LR, self.gamma)

  def remember(self, state, action, reward, nextState, isDone):
    self.memory.append((state, action, reward, nextState, isDone))

  def trainLongMemory(self):
    if len(self.memory) > BATCH_SIZE:
      miniSample = random.sample(self.memory, BATCH_SIZE)
    else:
      miniSample = self.memory
    
    states, actions, rewards, nextStates, isDones = zip(*miniSample)
    self.trainer.trainStep(states, actions, rewards, nextStates, isDones)

  def trainShortMemory(self, state, action, reward, nextState, isDone):
    self.trainer.trainStep(state, action, reward, nextState, isDone)

  def getAction(self, state):
    finalMove = [0, 0, 0]
    # dangerousDirectionsSum = state[0] + state[1] + state[2]
    
    if random.random() < self.epsilon and self.nGames < 100:
      move = random.randint(0, 2)
      finalMove[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      finalMove[move] = 1
    
    return finalMove
