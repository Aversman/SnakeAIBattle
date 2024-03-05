const playButtonElem        = document.querySelector('#play-ai')
const previewElem           = document.querySelector('.preview')
const connectionLoaderElem  = document.querySelector('.connect_loader')
const playButton            = document.querySelector('#play-btn')
const newGameButton         = document.querySelector('#new_game-btn')

// Snake directions: top, bottom, left, right

let game = new Game()

const socket = io()

// ms
const gameIterationDelay = 100

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
socket.on('ai_output_data', function(response) {
  setTimeout(() => {
    if (!game.isGameOver) {
      const inputAIData = game.gameStep(game.snake1.getSnakeDirection(response.snake1), game.snake2.getSnakeDirection(response.snake2))
      socket.emit('ai_input_data', inputAIData)
    }
  }, gameIterationDelay)
})

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
  if (!game.isGameOver) {
    const inputAIData = game.gameStep()
    socket.emit('ai_input_data', inputAIData)
  }
})

newGameButton.addEventListener('click', (event) => {
  event.preventDefault()
  game = new Game()
})