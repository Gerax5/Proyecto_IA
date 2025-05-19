import pygame
from snake_game import SnakeGame
from agent import Agent
import matplotlib.pyplot as plt
import pickle
import time

plt.ion()

def plot_scores(scores, mean_scores):
    plt.clf()
    plt.title('Puntaje por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Puntaje')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Promedio móvil', linestyle='--')
    plt.legend()
    plt.pause(0.1)


pygame.init()
font = pygame.font.SysFont('Arial', 25)
screen = pygame.display.set_mode((400, 400))

game = SnakeGame()
agent = Agent()

#cargar el modelo entrenado
with open('trained_model.pkl', 'rb') as f:
    agent.q_table = pickle.load(f)

scores = []
test_scores = []

agent.epsilon = 0.0  # sin exploracion, porque deberia de aplicar lo que sabe

for episode in range(10):
    state = game.reset()
    done = False
    total_reward = 0

    while not done: 
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        state = game.get_state()

        screen.fill((0, 0, 0))
        for part in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(part[0], part[1], game.block, game.block))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(game.food[0], game.food[1], game.block, game.block))
        pygame.display.flip()
        pygame.time.delay(50)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    scores.append(score)
    test_scores.append(sum(scores[-20:]) / min(len(scores), 20))  
    
    print(f"Episode {episode+1} | Score: {score} | Epsilon: {agent.epsilon:.3f}")
    print("Puntaje promedio en evaluación:", sum(test_scores)/len(test_scores))

    plot_scores(scores, test_scores)

pygame.quit()
plt.ioff()

time.sleep(2)
plt.show()
print("Entrenamiento finalizado.")
