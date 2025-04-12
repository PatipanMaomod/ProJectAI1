import json
import matplotlib.pyplot as plt

with open("rewards_per_episode.json", "r") as f:
    rewards = json.load(f)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()
