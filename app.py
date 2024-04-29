from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from time import sleep

import torch

from Agent import Agent
from Model import LinearQNetModel1, LinearQNetModel2

# $ flask --app app.py --debug run

app = Flask(__name__, static_folder='static/assets', template_folder='static/pages')
socketio = SocketIO(app)

app.config['SECRET_KEY'] = 'b57558e3-2a61-44fc-b338-f3d1febf2a56'

# ms
gameIterationDelay = 30

# Net
""" model1 = LinearQNetModel1(35, 28, 3)
model2 = LinearQNetModel1(35, 28, 3) """

model1 = LinearQNetModel2(85, 70, 3)
model2 = LinearQNetModel2(85, 70, 3)

model1.load_state_dict(torch.load("models/model2.pth"))
model2.load_state_dict(torch.load("models/model2.pth"))

# Agents
agent1 = Agent(model1, 0.1)
agent2 = Agent(model2, 0.1)

@app.route("/")
def home():
  return render_template('main.html')


@socketio.on('game_iteration')
def game_iteration_handler(state):
  snake1Action = agent1.getAction(state['snake1'])
  snake2Action = agent2.getAction(state['snake2'])
  agentsAction = {
    'snake1': snake1Action,
    'snake2': snake2Action
  }
  sleep(gameIterationDelay / 1000)
  emit('play_step', [state, agentsAction])


@socketio.on('game_reward')
def game_reward_handler(response):
  state, action, reward, newState, score, done = response
  agent1.trainShortMemory(state['snake1'], action['snake1'], reward['snake1'], newState['snake1'], done)
  agent2.trainShortMemory(state['snake2'], action['snake2'], reward['snake2'], newState['snake2'], done)

  agent1.remember(state['snake1'], action['snake1'], reward['snake1'], newState['snake1'], done)
  agent2.remember(state['snake2'], action['snake2'], reward['snake2'], newState['snake2'], done)

  if done:
    agent1.nGames += 1
    agent2.nGames += 1
    agent1.trainLongMemory()
    agent2.trainLongMemory()

    if (score['snake1'] > agent1.record):
      agent1.record = score['snake1']
      agent1.model.save()
    
    if (score['snake2'] > agent2.record):
      agent2.record = score['snake2']
      agent2.model.save()
    
    print('Game', agent1.nGames)
    print('[Agent1]', 'Score', score['snake1'], 'Record', agent1.record)
    print('[Agent2]', 'Score', score['snake2'], 'Record', agent2.record)
    emit('new_game')

  sleep(gameIterationDelay / 1000)
  emit('next_step_request')

if __name__ == '__main__':
  socketio.run(app, debug=True)
