from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit

# $ flask --app app.py --debug run
# /.venv/Scripts~ $ ./flask --app ../../app.py --debug run

app = Flask(__name__, static_folder='static/assets', template_folder='static/pages')
socketio = SocketIO(app)

app.config['SECRET_KEY'] = 'b57558e3-2a61-44fc-b338-f3d1febf2a56'

@app.route("/")
def home():
  return render_template('main.html')

@socketio.on('ai_input_data')
def ai_input_data_handler(message):
  print('recieved: ', message)
  response = {
    'snake1': [1, 0, 0],
    'snake2': [1, 0, 0]
  }
  emit('ai_output_data', response)

if __name__ == '__main__':
  socketio.run(app, debug=True)
