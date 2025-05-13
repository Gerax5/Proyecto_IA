import random
import numpy as np

class SnakeGame:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        self.block = 20
        self.reset()

    def reset(self):
        self.direction = 'RIGHT'
        self.head = [100, 100]
        self.snake = [self.head[:], [80, 100], [60, 100]]
        self.food = None
        self.place_food()
        self.score = 0
        self.frame = 0
        return self.get_state()

    def place_food(self):
        while True:
            x = random.randint(0, (self.w - self.block) // self.block) * self.block
            y = random.randint(0, (self.h - self.block) // self.block) * self.block
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def play_step(self, action): 
        self.frame += 1
        self.move(action)
        self.snake.insert(0, self.head[:])

        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def is_collision(self):
        if (self.head[0] < 0 or self.head[0] >= self.w or
            self.head[1] < 0 or self.head[1] >= self.h or
            self.head in self.snake[1:]):
            return True
        return False

    def move(self, action):
        directions = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  
            new_dir = directions[idx]
        elif np.array_equal(action, [0, 1, 0]): 
            new_dir = directions[(idx + 1) % 4]
        else: 
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head
        if self.direction == 'RIGHT':
            x += self.block
        elif self.direction == 'LEFT':
            x -= self.block
        elif self.direction == 'DOWN':
            y += self.block
        else:
            y -= self.block
        self.head = [x, y]

    def get_state(self):
        head = self.head
        point_l = [head[0] - self.block, head[1]]
        point_r = [head[0] + self.block, head[1]]
        point_u = [head[0], head[1] - self.block]
        point_d = [head[0], head[1] + self.block]

        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        state = [
            (dir_r and self.is_collision_at(point_r)) or 
            (dir_l and self.is_collision_at(point_l)) or 
            (dir_u and self.is_collision_at(point_u)) or 
            (dir_d and self.is_collision_at(point_d)),

            self.food[0] < self.head[0],  
            self.food[0] > self.head[0],  
            self.food[1] < self.head[1], 
            self.food[1] > self.head[1],  

            dir_l,
            dir_r,
            dir_u,
            dir_d
        ]
        return np.array(state, dtype=int)

    def is_collision_at(self, point):
        if (point[0] < 0 or point[0] >= self.w or
            point[1] < 0 or point[1] >= self.h or
            point in self.snake):
            return True
        return False
