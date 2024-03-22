from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from time import sleep

from Agent import Agent
from Model import LinearQNet, QTrainer

# $ flask --app app.py --debug run
# /.venv/Scripts~ $ ./flask --app ../../app.py --debug run

app = Flask(__name__, static_folder='static/assets', template_folder='static/pages')
socketio = SocketIO(app)

app.config['SECRET_KEY'] = 'b57558e3-2a61-44fc-b338-f3d1febf2a56'

# ms
gameIterationDelay = 1000

# Net
model = LinearQNet(40, 256, 3)
trainer = QTrainer(model, 1e-3, 0.9)

# Agents
agent1 = Agent(model, trainer)
agent2 = Agent(model, trainer)

@app.route("/")
def home():
  return render_template('main.html')


@socketio.on('game_iteration')
def game_iteration_handler(state):
  snake1Move = agent1.getAction(state['snake1'])
  snake2Move = agent2.getAction(state['snake2'])
  agentsDirection = {
    'snake1': snake1Move,
    'snake2': snake2Move
  }
  sleep(gameIterationDelay / 1000)
  emit('play_step', [state, agentsDirection])


@socketio.on('game_reward')
def game_reward_handler(response):
  oldState, direction, reward, state, score, isGameOver = response
  
  agent1.trainShortMemory(oldState['snake1'], direction['snake1'], reward['snake1'], state['snake1'], isGameOver)
  agent2.trainShortMemory(oldState['snake2'], direction['snake2'], reward['snake2'], state['snake2'], isGameOver)

  agent1.remember(oldState['snake1'], direction['snake1'], reward['snake1'], state['snake1'], isGameOver)
  agent2.remember(oldState['snake2'], direction['snake2'], reward['snake2'], state['snake2'], isGameOver)

  if isGameOver:
    emit('new_game')
    agent1.nGames += 1
    agent2.nGames += 1
    agent1.trainLongMemory()
    agent2.trainLongMemory()

    # Модель либо будет использоваться одна на двоих либо разные, посмотрим
    if (score['snake1'] > agent1.record):
      agent1.record = score['snake1']
      agent1.model.save()
    
    if (score['snake2'] > agent2.record):
      agent2.record = score['snake2']
      agent2.model.save()
    
    print('Game', agent1.nGames)
    print('[Agent1]', 'Score', score['snake1'], 'Record', agent1.record)
    print('[Agent2]', 'Score', score['snake2'], 'Record', agent2.record)

  sleep(gameIterationDelay / 1000)
  emit('next_step_request')

if __name__ == '__main__':
  socketio.run(app, debug=True)
