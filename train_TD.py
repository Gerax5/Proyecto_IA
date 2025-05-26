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

# pygame.init()
# screen = pygame.display.set_mode((400, 400))
# font = pygame.font.SysFont('Arial', 25)

game = SnakeGame()
agent = Agent()

scores = []
mean_scores = []

for episode in range(1000):
    state = game.reset()
    done = False
    trajectory = []  # Guardar (s, a, r) para cada paso

    while not done:
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        next_state = game.get_state()
        trajectory.append((state, action, reward))
        state = next_state

        # screen.fill((0, 0, 0))
        # for part in game.snake:
        #     pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(part[0], part[1], game.block, game.block))
        # pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(game.food[0], game.food[1], game.block, game.block))
        # pygame.display.flip()
        # pygame.time.delay(50)

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         exit()

    # Actualizar Q-table
    G = 0
    for state, action, reward in reversed(trajectory):
        G = reward + agent.gamma * G
        state = tuple(state)
        agent.ensure_state_exists(state)
        action_idx = agent.actions.index(action)
        agent.q_table[state][action_idx] += agent.lr * (G - agent.q_table[state][action_idx])

    scores.append(score)
    mean_scores.append(sum(scores[-20:]) / min(len(scores), 20))
    agent.epsilon = max(0.01, agent.epsilon * 0.99)
    print(f"Episode {episode+1} | Score: {score} | Epsilon: {agent.epsilon:.3f}")

    plot_scores(scores, mean_scores)

pygame.quit()
plt.ioff()
plt.show()

avg_score = sum(scores) / len(scores)
max_score = max(scores)
print("\nEstadísticas Finales:")
print(f"Promedio de puntaje: {avg_score:.2f}")
print(f"Puntaje más alto alcanzado: {max_score}")

with open('trained_model_TD.pkl', 'wb') as f:
    pickle.dump(agent.q_table, f)

print("Entrenamiento por refuerzo finalizado.")
