import pandas as pd
import matplotlib.pyplot as plt

def createGraph(agent1, agent2):
  n_games = len(agent1)
  average_score1 = sum(agent1) / n_games
  average_score2 = sum(agent2) / n_games
  df = pd.DataFrame({
    'Agent 1': agent1,
    'Agent 2': agent2
  })

  window_size = max(10, int(0.1 * n_games))
  
  rolling_agent1 = df['Agent 1'].rolling(window=window_size).mean()
  rolling_agent2 = df['Agent 2'].rolling(window=window_size).mean()
  plt.figure(figsize=(15, 7))
  plt.plot(df['Agent 1'], color='green', label='Агент 1', alpha=0.4)
  plt.plot(df['Agent 2'], color='red', label='Агент 2', alpha=0.4)
  rolling_agent1.plot(color='green', style='--')
  rolling_agent2.plot(color='red', style='--')
  
  plt.legend([f'Агент 1 (Средний счет: {average_score1:.2f})', f'Агент 2 (Средний счет: {average_score2:.2f})'])
  plt.xlabel('Номер игры')
  plt.ylabel('Счет')
  plt.savefig(f"graphs/graph_{n_games}.png")