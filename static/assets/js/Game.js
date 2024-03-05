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
    this.isGameOver = false
    this.foodMaxCount = 200
    this.foodCurCount = this.foodMaxCount
    this.arena  = new Arena()
    this.food   = new Food(this.arena.arenaWidth, this.arena.arenaHeight, this.arena.objectsWeight)
    this.snake1 = new Snake({
      arenaWidth: this.arena.arenaWidth,
      arenaHeight: this.arena.arenaHeight,
      snakeHeadColor: "#314647",
      snakeTailColor: "#497174",
      scoreElement: document.querySelector('#snake1-score-counter'),
      snakeTail: [
        [this.arena.objectsWeight * 3, this.arena.arenaHeight - this.arena.objectsWeight],
        [this.arena.objectsWeight * 2, this.arena.arenaHeight - this.arena.objectsWeight],
        [this.arena.objectsWeight * 1, this.arena.arenaHeight - this.arena.objectsWeight]
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
        [this.arena.arenaWidth - (this.arena.objectsWeight * 4), 0],
        [this.arena.arenaWidth - (this.arena.objectsWeight * 3), 0],
        [this.arena.arenaWidth - (this.arena.objectsWeight * 2), 0]
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

    if (isSnake1Hit) {
      gameReward.snake1 = -1
      this.isGameOver = true
    }
    if (isSnake2Hit) {
      gameReward.snake2 = -1
      this.isGameOver = true
    }
    if (isSnake1HitRival.value === true & isSnake2HitRival.value === false & !isSnake2Hit) {
      gameReward.snake2 = 1
    }
    if (isSnake2HitRival.value === true & isSnake1HitRival.value === false & !isSnake1Hit) {
      gameReward.snake1 = 1
    }
    if (this.snake1.isSnakeAteFood([this.food.foodX, this.food.foodY], this.snake2.snakeTail)) {
      gameReward.snake1 = 1
      this._foodGenerate(this.snake1.snakeTail, this.snake2.snakeTail)
    }
    if (this.snake2.isSnakeAteFood([this.food.foodX, this.food.foodY], this.snake1.snakeTail)) {
      gameReward.snake2 = 1
      this._foodGenerate(this.snake1.snakeTail, this.snake2.snakeTail)
    }
    this._updateScreen()
    // for debug
    console.log(gameReward)
    return this._getInputDataAI(gameReward)
  }
  /*
    Выполняет поиск по заданной стороне для указанной змейки;
    В forPlayer подается номер игрока (1 или 2), относительно которого будет производиться поиск;
    direction принимает значения: topLeft, topRight, bottomLeft, bottomRight, left, right, top, bottom;
  */
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
    const result = [0, 0, 0, 0, 0]
    let distance = 1
    let isFinded = false
    
    while((curX > 0 & curX < this.arena.arenaWidth) & (curY > 0 & curY < this.arena.arenaHeight)) {
      if (isFinded) {
        break
      }
      if (curX === this.food.foodX & curY === this.food.foodY) {
        isFinded = true
        result[0] = 1 / distance
        result[1] = 1
        break
      }
      for (let i = 0; i < snakeMain.snakeTail.length; i++) {
        if (snakeMain.snakeTail[i][0] === curX & snakeMain.snakeTail[i][1] === curY) {
          isFinded = true
          result[0] = 1 / distance
          result[2] = 1
          break
        }
      }
      for (let i = 0; i < snakeRival.snakeTail.length; i++) {
        if (snakeRival.snakeTail[i][0] === curX & snakeRival.snakeTail[i][1] === curY) {
          isFinded = true
          result[0] = 1 / distance
          result[3] = 1
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
      result[0] = 1 / distance
      result[4] = 1
    }
    result[0] = Number(result[0].toFixed(3))
    return result
  }
  _getInputDataAI(reward) {
    const snake1Data = [
      ...this._findObject(1, 'topLeft'),
      ...this._findObject(1, 'top'),
      ...this._findObject(1, 'topRight'),
      ...this._findObject(1, 'right'),
      ...this._findObject(1, 'bottomRight'),
      ...this._findObject(1, 'bottom'),
      ...this._findObject(1, 'bottomLeft'),
      ...this._findObject(1, 'left'),

      this.snake1.snakeScore / this.foodMaxCount,
      this.snake2.snakeScore / this.foodMaxCount,
      this.foodCurCount / (this.foodMaxCount - 1),
      reward.snake1
    ]
    const snake2Data = [
      ...this._findObject(2, 'topLeft'),
      ...this._findObject(2, 'top'),
      ...this._findObject(2, 'topRight'),
      ...this._findObject(2, 'right'),
      ...this._findObject(2, 'bottomRight'),
      ...this._findObject(2, 'bottom'),
      ...this._findObject(2, 'bottomLeft'),
      ...this._findObject(2, 'left'),

      this.snake2.snakeScore / this.foodMaxCount,
      this.snake1.snakeScore / this.foodMaxCount,
      this.foodCurCount / (this.foodMaxCount - 1),
      reward.snake2
    ]
    return {
      "snake1": snake1Data,
      "snake2": snake2Data
    }
  }
  _foodGenerate(snakeTail1, snakeTail2) {
    if (this.foodCurCount === 0) {
      this.isGameOver = true
      return false
    }
    this.foodCurCount--
    this.foodCounterElem.innerHTML = this.foodCurCount
    this.food.foodGenerate(snakeTail1, snakeTail2)
    return true
  }
  _updateScreen() {
    this.arena.gridClear()
    this.arena.snakeRender(this.snake1.snakeHeadColor, this.snake1.snakeColor, this.snake1.snakeTail)
    this.arena.snakeRender(this.snake2.snakeHeadColor, this.snake2.snakeColor, this.snake2.snakeTail)
    this.arena.foodRender(this.food.foodColor, this.food.foodX, this.food.foodY)
  }
}

/*
 Нормализованная структура входных данных для сети;
 Рассматривается все относительно головы;
 Голова - 4 вершин, 8 сторон. Первой вершиной считаем верхнюю левую вершину, далее по часовой идут последующие стороны;
 Кодировка для типов объектов каждой стороны:
 [Расстояние до объекта, Яблоко, Свой хвост, Чужой хвост, Стена];
 Пример:
 [0.6, 0, 0, 0, 0, 1] - Значит, что до стенки 0.6 расстояния;
 
 Нормализированный итоговый входной вектор по всем сторонам + доп информация:
 [
  0.6, 0, 0, 0, 0, 1, # для 0 вершины
  0.2, 0, 1, 0, 0, 0, # для 1 вершины
  # .... также для остальных вершин
  # далее передаются дополнительные данные
  0.4, # Нормализованный счет змейки (текущий счет / максимально достижимый счет)
  0.3, # Нормализованный счет змейки-соперника
  0.5, # Оставшееся количество генерируемых яблок (текущее количество / начальное количество)
  0 # награда за действие: {-1, 0, 1}
 ]
*/

/*
 На выходе ожидаем вектор длины три: [продолжить направление, повернуть влево, повернуть вправо];
 берется максимум из расспределения вероятности для конкретного решения направления;
*/