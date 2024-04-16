class Arena {
  constructor() {
    this.canvas = document.querySelector('#canvas')
    this.ctx = canvas.getContext("2d")
    this.arenaWidth = Number(this.canvas.width)
    this.arenaHeight = Number(this.canvas.height)
    this.objectsWeight = 20
    this.allArenaPoints = []

    for (let i = 0; i < (this.arenaWidth / this.objectsWeight); i++) {
      for (let j = 0; j < (this.arenaHeight / this.objectsWeight); j++) {
        this.allArenaPoints.push([i * this.objectsWeight, j * this.objectsWeight]);
      }
    }
    this._gridInit()
  }
  _gridInit() {
    const grid = document.querySelector('#game-grid')
    grid.style.width = `${this.arenaWidth}px`
    grid.style.height = `${this.arenaHeight}px`
    grid.style.gridTemplateColumns = `repeat(${this.arenaWidth / this.objectsWeight}, 1fr)`
    grid.style.gridTemplateRows = `repeat(${this.arenaHeight / this.objectsWeight}, 1fr)`
    for (let i = 0; i < ((this.arenaWidth / this.objectsWeight) * (this.arenaHeight / this.objectsWeight)); i++) {
      const div = document.createElement('div')
      grid.append(div)
    }
  }

  gridClear() {
    this.ctx.clearRect(0, 0, this.arenaWidth, this.arenaHeight)
  }

  foodRender(foodColor, foodX, foodY) {
    this.ctx.fillStyle = foodColor
    this.ctx.fillRect(foodX + (this.objectsWeight / 4), foodY + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
  }
  
  snakeRender(snakeHeadColor, snakeTailColor, snakeTail) {
    snakeTail.forEach((elem, index) => {
      if (index === 0) {
        this.ctx.fillStyle = snakeHeadColor
        this.ctx.fillRect(elem[0] + (this.objectsWeight / 4), elem[1] + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
      }else {
        this.ctx.fillStyle = snakeTailColor
        this.ctx.fillRect(elem[0] + (this.objectsWeight / 4), elem[1] + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
      }
      if (index !== 0) {
        if (Math.abs(elem[1] - snakeTail[index - 1][1]) === 0) {
          if (elem[0] - snakeTail[index - 1][0] < 0) {
            this.ctx.fillStyle = this.snakeColor
            this.ctx.fillRect((elem[0] + this.objectsWeight / 2) + (this.objectsWeight / 4), (elem[1]) + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
          }else {
            this.ctx.fillStyle = this.snakeColor
            this.ctx.fillRect((elem[0] - this.objectsWeight / 2) + (this.objectsWeight / 4), (elem[1]) + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
          }
        }else if (Math.abs(elem[0] - snakeTail[index - 1][0]) === 0) {
          if (elem[1] - snakeTail[index - 1][1] < 0) {
            this.ctx.fillStyle = this.snakeColor
            this.ctx.fillRect((elem[0]) + (this.objectsWeight / 4), (elem[1] + this.objectsWeight / 2) + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
          }else {
            this.ctx.fillStyle = this.snakeColor
            this.ctx.fillRect((elem[0]) + (this.objectsWeight / 4), (elem[1] - this.objectsWeight / 2) + (this.objectsWeight / 4), this.objectsWeight / 2, this.objectsWeight / 2)
          }
        }
      }
    })
  }
}

class Food {
  constructor(arenaWidth, arenaHeight, objectsWeight) {
    this.foodColor = '#bf323b'
    this.foodX = 0
    this.foodY = 0
    this.arenaWidth = arenaWidth
    this.arenaHeight = arenaHeight
    this.objectsWeight = objectsWeight
  }
  foodGenerate(snakeTail1, snakeTail2) {
    const foodX = Math.floor(Math.random() * (this.arenaWidth / this.objectsWeight)) * this.objectsWeight
    const foodY = Math.floor(Math.random() * (this.arenaHeight / this.objectsWeight)) * this.objectsWeight
    let flag1 = false
    let flag2 = false
    for (let i = 0; i < snakeTail1.length; i++) {
      if (foodX === snakeTail1[i][0] && foodY === snakeTail1[i][1]) {
        flag1 = true
        break
      }
    }
    for (let i = 0; i < snakeTail2.length; i++) {
      if (foodX === snakeTail2[i][0] && foodY === snakeTail2[i][1]) {
        flag2 = true
        break
      }
    }

    if (flag1 | flag2) {
      this.foodGenerate(snakeTail1, snakeTail2)
    }else {
      this.foodX = foodX
      this.foodY = foodY
    }
  }
}

class Snake {
  constructor(snakeConfig) {
    this.arenaWidth = snakeConfig.arenaWidth
    this.arenaHeight = snakeConfig.arenaHeight
    this.snakeColor = snakeConfig.snakeTailColor // '#497174'
    this.snakeHeadColor = snakeConfig.snakeHeadColor // '#314647'
    this.snakeHeadX = snakeConfig.snakeTail[0][0]
    this.snakeHeadY = snakeConfig.snakeTail[0][1]
    this.snakeTail = snakeConfig.snakeTail
    this.snakeDirection = snakeConfig.snakeDirection
    this.scoreElement = snakeConfig.scoreElement
    this.scoreElement.innerHTML = 0
    this.snakeScore = 0
  }
  snakeMove(objectsWeight) {
    if (this.snakeDirection === 'right') {
      this.snakeHeadX += objectsWeight
      this.snakeTail.pop()
      this.snakeTail.unshift([this.snakeHeadX, this.snakeHeadY])
    }
    else if (this.snakeDirection === 'left') {
      this.snakeHeadX -= objectsWeight
      this.snakeTail.pop()
      this.snakeTail.unshift([this.snakeHeadX, this.snakeHeadY])
    }
    else if (this.snakeDirection === 'top') {
      this.snakeHeadY -= objectsWeight
      this.snakeTail.pop()
      this.snakeTail.unshift([this.snakeHeadX, this.snakeHeadY])
    }
    else if (this.snakeDirection === 'bottom') {
      this.snakeHeadY += objectsWeight
      this.snakeTail.pop()
      this.snakeTail.unshift([this.snakeHeadX, this.snakeHeadY])
    }
  }
  isSnakeAteFood(foodCoords) {
    if (this.snakeHeadX === foodCoords[0] && this.snakeHeadY === foodCoords[1]) {
      this.snakeAddPoint()
      this.snakeScore++
      this.scoreElement.innerHTML = this.snakeScore
      return true
    }
    return false
  }
  snakeAddPoint() {
    const lastSnakePoint = this.snakeTail[this.snakeTail.length - 1]
    const secondLastSnakePoint = this.snakeTail[this.snakeTail.length - 2]
    let x = lastSnakePoint[0]
    let y = lastSnakePoint[1]
    if (lastSnakePoint[0] === secondLastSnakePoint[0]) {
      y = lastSnakePoint[1] + (lastSnakePoint[1] - secondLastSnakePoint[1])
    }else if (lastSnakePoint[1] === secondLastSnakePoint[1]) {
      x = lastSnakePoint[0] + (lastSnakePoint[0] - secondLastSnakePoint[0])
    }
    this.snakeTail.push([x, y])
  }
  // isHitRival - неявный объект, которыйменяет значение value на true, если змейка столкнулась об соперника
  isSnakeHit(snake2Tail, isHitRival = null) {
    if (this.snakeHeadX < 0 || this.snakeHeadX >= this.arenaWidth) {
      return true
    }
    if (this.snakeHeadY < 0 || this.snakeHeadY >= this.arenaHeight) {
      return true
    }
    for (let i = 1; i < this.snakeTail.length; i++) {
      if (this.snakeHeadX === this.snakeTail[i][0] && this.snakeHeadY === this.snakeTail[i][1]) {
        return true
      }
    }
    for (let i = 0; i < snake2Tail.length; i++) {
      if (this.snakeHeadX === snake2Tail[i][0] && this.snakeHeadY === snake2Tail[i][1]) {
        isHitRival.value = true
        return true
      }
    }
    return false
  }
  snakeChangeDirection(direction) {
    if (this.snakeDirection === 'top' && direction === 'bottom') {
      return
    }
    if (this.snakeDirection === 'bottom' && direction === 'top') {
      return
    }
    if (this.snakeDirection === 'left' && direction === 'right') {
      return
    }
    if (this.snakeDirection === 'right' && direction === 'left') {
      return
    }
    this.snakeDirection = direction
  }
  // outputArray - выходной массив сети, имеющее значение: [продолжить направление, повернуть влево, повернуть вправо]
  getSnakeDirection(outputArray) {
    if (outputArray.length !== 3) return null
    // продолжить движение
    if (outputArray[0] === 1) {
      return this.snakeDirection
    }
    // повернуть влево
    if (outputArray[1] === 1) {
      switch (this.snakeDirection) {
        case 'top':
          return 'left'
        case 'right':
          return 'top'
        case 'bottom':
          return 'right'
        case 'left':
          return 'bottom'
        default:
          return null
      }
    }
    // повернуть вправо
    if (outputArray[2] === 1) {
      switch (this.snakeDirection) {
        case 'top':
          return 'right'
        case 'right':
          return 'bottom'
        case 'bottom':
          return 'left'
        case 'left':
          return 'top'
        default:
          return null
      }
    }
    return null
  }
}

class Game {
  constructor() {
    // счетчик бесполезных ходов
    this.gameFreeIteration = 0
    this.isGameOver = 0
    this.foodSpawnCount = 6
    this.foodMaxCount = 200
    this.foodCurCount = this.foodMaxCount
    this.arena = new Arena()
    this.food = []
    for (let i = 0; i < this.foodSpawnCount; i++) {
      this.food.push(new Food(this.arena.arenaWidth, this.arena.arenaHeight, this.arena.objectsWeight))
    }

    this.snake1 = new Snake({
      arenaWidth: this.arena.arenaWidth,
      arenaHeight: this.arena.arenaHeight,
      snakeHeadColor: "#314647",
      snakeTailColor: "#497174",
      scoreElement: document.querySelector('#snake1-score-counter'),
      snakeTail: [
        [this.arena.objectsWeight * 20, this.arena.arenaHeight - (this.arena.objectsWeight * 14)],
        [this.arena.objectsWeight * 19, this.arena.arenaHeight - (this.arena.objectsWeight * 14)]
    ],
      snakeDirection: "right"
    })
    this.snake2 = new Snake({
      arenaWidth: this.arena.arenaWidth,
      arenaHeight: this.arena.arenaHeight,
      snakeHeadColor: "#473131",
      snakeTailColor: "#744949",
      scoreElement: document.querySelector('#snake2-score-counter'),
      snakeTail: [
        [this.arena.arenaWidth - (this.arena.objectsWeight * 20), (this.arena.objectsWeight * 14)],
        [this.arena.arenaWidth - (this.arena.objectsWeight * 19), (this.arena.objectsWeight * 14)]
    ],
      snakeDirection: "left"
    })
    this.foodCounterElem = document.querySelector('#game-toolbar-score-counter')
    this.foodCounterElem.innerHTML = this.foodCurCount
    this._foodGenerate(this.snake1.snakeTail, this.snake2.snakeTail)
    this._updateScreen()
  }
  gameStep(snake1Direction=null, snake2Direction=null) {
    if (this.isGameOver) {
      return null
    }
    const gameReward = {
      snake1: 0,
      snake2: 0
    }
    if (snake1Direction) {
      this.snake1.snakeChangeDirection(snake1Direction)
    }
    if (snake2Direction) {
      this.snake2.snakeChangeDirection(snake2Direction)
    }
    
    this.snake1.snakeMove(this.arena.objectsWeight)
    this.snake2.snakeMove(this.arena.objectsWeight)

    // Дополнительно проверяем, ударились ли они друг об друга, чтобы задать reward конкурируемым
    const isSnake1HitRival = {value: false}
    const isSnake2HitRival = {value: false}
    const isSnake1Hit = this.snake1.isSnakeHit(this.snake2.snakeTail, isSnake1HitRival)
    const isSnake2Hit = this.snake2.isSnakeHit(this.snake1.snakeTail, isSnake2HitRival)

    for (let i = 0; i < this.foodSpawnCount; i++) {
      if (this.snake1.isSnakeAteFood([this.food[i].foodX, this.food[i].foodY])) {
        gameReward.snake1 = 15
        this.gameFreeIteration = 0
        this._foodGenerate(this.snake1.snakeTail, this.snake2.snakeTail, i)
        break
      }
    }
    
    for (let i = 0; i < this.foodSpawnCount; i++) {
      if (this.snake2.isSnakeAteFood([this.food[i].foodX, this.food[i].foodY])) {
        gameReward.snake2 = 15
        this.gameFreeIteration = 0
        this._foodGenerate(this.snake1.snakeTail, this.snake2.snakeTail, i)
        break
      }
    }

    
    // проверка на зацикливание, если агент тупит долгое время, обнуляем игру
    if (gameReward.snake1 === 0 & gameReward.snake2 === 0) {
      this.gameFreeIteration++
    }

    if (this.gameFreeIteration > 300) {
      gameReward.snake1 = -5
      gameReward.snake2 = -5
    }

    if (isSnake1Hit) {
      gameReward.snake1 = -10
      this.isGameOver = 1
    }
    if (isSnake2Hit) {
      gameReward.snake2 = -10
      this.isGameOver = 1
    }

    if (this.gameFreeIteration > 600) {
      gameReward.snake1 = -20
      gameReward.snake2 = -20
      this.gameFreeIteration = 0
      this.isGameOver = 1
    }

    this._updateScreen()

    // for debug
    console.log(gameReward)
    console.log(this.gameFreeIteration)
    return gameReward
  }
  /*
    Выполняет поиск по заданной стороне для указанной змейки;
    В forPlayer подается номер игрока (1 или 2), относительно которого будет производиться поиск;
    direction принимает значения: topLeft, topRight, bottomLeft, bottomRight, left, right, top, bottom;
  */
  // Возвращает массив вида: [Яблоко, Свой хвост, Чужой хвост, Стена]
  _findObject(forPlayer, direction) {
    const snakeMain = (forPlayer === 1) ? this.snake1 : this.snake2
    const snakeRival = (forPlayer === 1) ? this.snake2 : this.snake1
    const objectsWeight = this.arena.objectsWeight
    let curX = snakeMain.snakeHeadX
    let curY = snakeMain.snakeHeadY
    switch (direction) {
      case 'top':
        curY -= objectsWeight
        break;
      case 'bottom':
        curY += objectsWeight
        break;
      case 'left':
        curX -= objectsWeight
        break;
      case 'right':
        curX += objectsWeight
        break;
      case 'topLeft':
        curX -= objectsWeight
        curY -= objectsWeight
        break;
      case 'topRight':
        curX += objectsWeight
        curY -= objectsWeight
        break;
      case 'bottomLeft':
        curX -= objectsWeight
        curY += objectsWeight
        break;
      case 'bottomRight':
        curX += objectsWeight
        curY += objectsWeight
        break;
      default:
        break;
    }
    const result = [0, 0, 0, 0]
    let distance = 1
    let isFinded = false
    
    while((curX > 0 & curX < this.arena.arenaWidth) & (curY > 0 & curY < this.arena.arenaHeight)) {
      if (isFinded) {
        break
      }
      for (let i = 0; i < this.foodSpawnCount; i++) {
        if (this.food[i].foodX === curX & this.food[i].foodY === curY) {
          isFinded = true
          result[0] = 1
          // result[4] = 1 / distance
          break
        }
      }
      for (let i = 0; i < snakeMain.snakeTail.length; i++) {
        if (snakeMain.snakeTail[i][0] === curX & snakeMain.snakeTail[i][1] === curY) {
          isFinded = true
          result[1] = 1
          // result[4] = 1 / distance
          break
        }
      }
      for (let i = 0; i < snakeRival.snakeTail.length; i++) {
        if (snakeRival.snakeTail[i][0] === curX & snakeRival.snakeTail[i][1] === curY) {
          isFinded = true
          result[2] = 1
          // result[4] = 1 / distance
          break
        }
      }
      distance++
      switch (direction) {
        case 'top':
          curY -= objectsWeight
          break;
        case 'bottom':
          curY += objectsWeight
          break;
        case 'left':
          curX -= objectsWeight
          break;
        case 'right':
          curX += objectsWeight
          break;
        case 'topLeft':
          curX -= objectsWeight
          curY -= objectsWeight
          break;
        case 'topRight':
          curX += objectsWeight
          curY -= objectsWeight
          break;
        case 'bottomLeft':
          curX -= objectsWeight
          curY += objectsWeight
          break;
        case 'bottomRight':
          curX += objectsWeight
          curY += objectsWeight
          break;
        default:
          break;
      }
    }

    if (!isFinded) {
      // result[4] = 1 / distance
      result[3] = 1
    }
    // result[4] = Number(result[4].toFixed(3))
    return result
  }
  // Проверяет вокруг головы змейки все препятствия и возвращает массив опасных шагов
  // Возвращает массив вида: [Продолжить направление, Слева, Справа]
  _findDangerDirections(forPlayer) {
    const result  = [0, 0, 0]
    const snake   = (forPlayer === 1) ? this.snake1 : this.snake2
    const snake2  = (forPlayer === 1) ? this.snake2 : this.snake1
    if (snake.snakeDirection === 'top') {
      if (snake.snakeHeadY === 0) {
        result[0] = 1
      }
      if (snake.snakeHeadX === 0) {
        result[1] = 1
      }
      if (snake.snakeHeadX === this.arena.arenaWidth - this.arena.objectsWeight) {
        result[2] = 1
      }
      snake.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[2] = 1
        }
      })
      snake2.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[2] = 1
        }
      })
    }
    if (snake.snakeDirection === 'bottom') {
      if (snake.snakeHeadY === this.arena.arenaHeight - this.arena.objectsWeight) {
        result[0] = 1
      }
      if (snake.snakeHeadX === this.arena.arenaWidth - this.arena.objectsWeight) {
        result[1] = 1
      }
      if (snake.snakeHeadX === 0) {
        result[2] = 1
      }
      snake.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[2] = 1
        }
      })
      snake2.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[2] = 1
        }
      })
    }
    if (snake.snakeDirection === 'left') {
      if (snake.snakeHeadX === 0) {
        result[0] = 1
      }
      if (snake.snakeHeadY === this.arena.arenaHeight - this.arena.objectsWeight) {
        result[1] = 1
      }
      if (snake.snakeHeadY === 0) {
        result[2] = 1
      }
      snake.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[2] = 1
        }
      })
      snake2.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX - this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[2] = 1
        }
      })
    }
    if (snake.snakeDirection === 'right') {
      if (snake.snakeHeadX === this.arena.arenaWidth - this.arena.objectsWeight) {
        result[0] = 1
      }
      if (snake.snakeHeadY === 0) {
        result[1] = 1
      }
      if (snake.snakeHeadY === this.arena.arenaHeight - this.arena.objectsWeight) {
        result[2] = 1
      }
      snake.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[2] = 1
        }
      })
      snake2.snakeTail.forEach((tail) => {
        if (tail[0] === snake.snakeHeadX + this.arena.objectsWeight & tail[1] === snake.snakeHeadY) {
          result[0] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY - this.arena.objectsWeight) {
          result[1] = 1
        }
        if (tail[0] === snake.snakeHeadX & tail[1] === snake.snakeHeadY + this.arena.objectsWeight) {
          result[2] = 1
        }
      })
    }
    return result
  }
  getAgentsState() {
    const snake1Data = [
      ...this._findDangerDirections(1),
      ...this._findObject(1, 'topLeft'),
      ...this._findObject(1, 'top'),
      ...this._findObject(1, 'topRight'),
      ...this._findObject(1, 'right'),
      ...this._findObject(1, 'bottomRight'),
      ...this._findObject(1, 'bottom'),
      ...this._findObject(1, 'bottomLeft'),
      ...this._findObject(1, 'left'),
    ]
    const snake2Data = [
      ...this._findDangerDirections(2),
      ...this._findObject(2, 'topLeft'),
      ...this._findObject(2, 'top'),
      ...this._findObject(2, 'topRight'),
      ...this._findObject(2, 'right'),
      ...this._findObject(2, 'bottomRight'),
      ...this._findObject(2, 'bottom'),
      ...this._findObject(2, 'bottomLeft'),
      ...this._findObject(2, 'left'),
    ]
    return {
      "snake1": snake1Data,
      "snake2": snake2Data
    }
  }
  _foodGenerate(snakeTail1, snakeTail2, foodIdx = null) {
    if (this.foodCurCount === 0) {
      this.isGameOver = 1
      return false
    }
    if (foodIdx !== null) {
      this.foodCurCount--
      this.foodCounterElem.innerHTML = this.foodCurCount
      this.food[foodIdx].foodGenerate(snakeTail1, snakeTail2)
      return true
    }
    for (let i = 0; i < this.foodSpawnCount; i++) {
      this.foodCurCount--
      this.foodCounterElem.innerHTML = this.foodCurCount
      this.food[i].foodGenerate(snakeTail1, snakeTail2)
    }
    return true
  }
  _updateScreen() {
    this.arena.gridClear()
    this.arena.snakeRender(this.snake1.snakeHeadColor, this.snake1.snakeColor, this.snake1.snakeTail)
    this.arena.snakeRender(this.snake2.snakeHeadColor, this.snake2.snakeColor, this.snake2.snakeTail)
    this.food.forEach(food => {
      this.arena.foodRender(food.foodColor, food.foodX, food.foodY)
    })
  }
}