import pygame
from snake_game import SnakeGame
from agent import Agent
import matplotlib.pyplot as plt
import pickle 

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

scores = []
mean_scores = []

for episode in range(200):
    state = game.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        next_state = game.get_state()
        agent.train_short_memory(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

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
    mean_scores.append(sum(scores[-20:]) / min(len(scores), 20))  
    agent.epsilon = max(0.01, agent.epsilon * 0.99)
    print(f"Episode {episode+1} | Score: {score} | Epsilon: {agent.epsilon:.3f}")

    plot_scores(scores, mean_scores)

pygame.quit()
plt.ioff()
plt.show()
print("Entrenamiento finalizado.")


# Guardar el modelo entrenado
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(agent.q_table, f)
