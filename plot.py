import json
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# โหลดข้อมูล metrics จากไฟล์
with open("ppo_metrics.json", "r") as f:
    metrics = json.load(f)

# Smooth ค่า reward เพื่อให้ดู trend ชัดเจนขึ้น
smoothed_rewards = uniform_filter1d(metrics["rewards_per_episode"], size=10)
actor_loss_per_episode = uniform_filter1d(metrics["actor_loss_per_episode"], size=100)
critic_loss_per_episode = uniform_filter1d(metrics["critic_loss_per_episode"], size=100)
# สร้างกราฟ
plt.figure(figsize=(16, 10))
plt.plot(smoothed_rewards, label="Smoothed Reward", color="blue")
plt.plot(actor_loss_per_episode, label="Actor Loss", color="orange")
plt.plot(metrics["critic_loss_per_episode"], label="Critic Loss", color="green")
plt.xlabel("Episode")
plt.ylim(-100, 1000)
plt.ylabel("Value")
plt.title("PPO Training Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
