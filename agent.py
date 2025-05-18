import numpy as np
import random

class Agent:
    def __init__(self):
        self.q_table = {} 
        self.gamma = 0.9      
        self.epsilon = 1.0    
        self.lr = 0.1         

        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def get_action(self, state):
        state = tuple(state)
        self.ensure_state_exists(state)

        if random.random() < self.epsilon:
            # Explorar acción aleatoria
            return random.choice(self.actions)
        else:
            # Explotar mejor acción conocida
            best_idx = np.argmax(self.q_table[state])
            return self.actions[best_idx]

    def ensure_state_exists(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]

    #Actualizar valor Q(s,a)
    def train_short_memory(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        # Valor estimado de la acción actual
        max_future_q = max(self.q_table[next_state])

        target_q = reward if done else reward + self.gamma * max_future_q

        # Actualizar valor Q con fórmula Q-learning
        self.q_table[state][action_idx] += self.lr * (target_q - current_q)

    def remember(self, state, action, reward, next_state, done):
        self.train_short_memory(state, action, reward, next_state, done)