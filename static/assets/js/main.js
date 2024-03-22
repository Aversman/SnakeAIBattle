const playButtonElem        = document.querySelector('#play-ai')
const previewElem           = document.querySelector('.preview')
const connectionLoaderElem  = document.querySelector('.connect_loader')
const playButton            = document.querySelector('#play-btn')
const newGameButton         = document.querySelector('#new_game-btn')

// Snake directions: top, bottom, left, right

let game = new Game()

const socket = io()

// SocketIO events
socket.on("connect", () => {
  console.log('WS Connected')
  connectionLoaderElem.style.visibility = 'hidden'
  connectionLoaderElem.style.opacity = '0'
})

socket.on("disconnect", () => {
  console.log('WS Disconnected')
  connectionLoaderElem.style.visibility = 'visible'
  connectionLoaderElem.style.opacity = '1'
})

// game iteration
socket.on('play_step', function(response) {
  const state = response[0]
  const direction = response[1]
  const reward = game.gameStep(game.snake1.getSnakeDirection(direction.snake1), game.snake2.getSnakeDirection(direction.snake2))
  const score = {'snake1': game.snake1.snakeScore, 'snake2': game.snake2.snakeScore}
  socket.emit('game_reward', [state, direction, reward, game.getAgentsState(), score, game.isGameOver])
})

socket.on('new_game', function() {
  game = new Game()
})

socket.on('next_step_request', function() {
  socket.emit('game_iteration', game.getAgentsState())
})

// DOM events
document.addEventListener("DOMContentLoaded", () => {
  playButtonElem.addEventListener('click', (event) => {
    event.preventDefault()
    previewElem.style.visibility = 'hidden'
    previewElem.style.opacity = '0'
    startGameEvent = true
  })
})

// init game event
playButton.addEventListener('click', (event) => {
  event.preventDefault()
  socket.emit('game_iteration', game.getAgentsState())
  playButton.disabled = true
  newGameButton.disabled = true
})

newGameButton.addEventListener('click', (event) => {
  event.preventDefault()
  if (game.isGameOver) {
    game = new Game()
  }
})