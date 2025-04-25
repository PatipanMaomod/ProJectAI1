import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import json
import asyncio
import websockets
import uuid
import os


# === PPO Configuration ===
class PPOConfig:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.clip_ratio = 0.2
        self.update_epochs = 4
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.buffer_size = 100000


# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, state_dim, max_size=10000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.bool_)
        self.log_probs = np.zeros((max_size,), dtype=np.float32)
        self.values = np.zeros((max_size + 1,), dtype=np.float32)
        self.returns = np.zeros((max_size,), dtype=np.float32)
        self.advantages = np.zeros((max_size,), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, state, action, reward, next_state, done, log_prob=None, value=None, next_value=None):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        if log_prob is not None:
            self.log_probs[self.ptr] = log_prob
        if value is not None:
            self.values[self.ptr] = value
        if next_value is not None:
            self.values[self.ptr + 1] = next_value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def compute_gae(self, gamma=0.99, lam=0.95):
        gae = 0.0
        for t in reversed(range(self.size)):
            delta = self.rewards[t] + gamma * (1 - self.dones[t]) * self.values[t + 1] - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def to_tensor(self):
        return dict(
            states=torch.tensor(self.states[:self.size], dtype=torch.float32, device=self.device),
            actions=torch.tensor(self.actions[:self.size], dtype=torch.long, device=self.device),
            rewards=torch.tensor(self.rewards[:self.size], dtype=torch.float32, device=self.device),
            next_states=torch.tensor(self.next_states[:self.size], dtype=torch.float32, device=self.device),
            dones=torch.tensor(self.dones[:self.size], dtype=torch.float32, device=self.device),
            log_probs=torch.tensor(self.log_probs[:self.size], dtype=torch.float32, device=self.device),
            returns=torch.tensor(self.returns[:self.size], dtype=torch.float32, device=self.device),
            advantages=torch.tensor(self.advantages[:self.size], dtype=torch.float32, device=self.device),
        )

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.next_states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(False)
        self.log_probs.fill(0)
        self.values.fill(0)
        self.returns.fill(0)
        self.advantages.fill(0)


# === Actor-Critic Networks ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# === State Wrapper ===
class Godot:
    def __init__(self, data):

        self.agent, self.goal , self.distance, self.vector_to_goal , self.velocity, self.see_goal= data["agent"], data["goal"], data["distance"], data["vector_to_goal"] ,data["velocitys"],data["see_goal"]
        self.state = np.array(self.agent + self.goal + self.distance + self.vector_to_goal + self.velocity + self.see_goal, dtype=np.float32)
    def get_state(self):
        return self.state

    def get_dim(self):
        return len(self.state), 4


# === Utility Functions ===
def save_model(actor, critic, path="model/actor.pth", critic_path="model/critic.pth", info_path="model/info.txt"):
    # บันทึก actor และ critic แยกไฟล์
    torch.save(actor.state_dict(), path)
    torch.save(critic.state_dict(), critic_path)

    # เขียนข้อมูลเพิ่มเติม เช่น uuid หรือ timestamp
    with open(info_path, "w") as f:
        f.write(f"Saved model with UUID: {uuid.uuid4()}\n")


def load_model(actor, critic, actor_path="model/actor.pth", critic_path="model/critic.pth"):
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path))
        print(f"✅ Actor loaded from {actor_path}")
    else:
        print(f"🔸 No actor model found at {actor_path}")

    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path))
        print(f"✅ Critic loaded from {critic_path}")
    else:
        print(f"🔸 No critic model found at {critic_path}")


async def receive_from_godot(websocket):
    try:
        message = await websocket.recv()
        return json.loads(message)
    except json.JSONDecodeError:
        return None


async def send_to_godot(websocket, action):
    await websocket.send(json.dumps(str(action)))


# === PPO Training via WebSocket ===
async def echo(websocket, actor, critic, actor_optimizer, critic_optimizer, buffer, config):
    print("📡 Connected to client")
    await websocket.send('Statring PPO Training...')
    device = buffer.device
    prev_state = prev_value = prev_log_prob = prev_action = None
    actor_loss = critic_loss = 0.0
    episode_reward = 0
    episode = 0
    rewards_per_episode = []
    actor_loss_per_episode = []
    critic_loss_per_episode = []

    try:
        while episode < 1000:
            episode += 1
            steps = 0
            while True:
                steps += 1
                data = await receive_from_godot(websocket)
                if data is None:
                    continue


                reward, done , agent_die = data["reward"], data["done"] ,data["agent_is_die"]
                episode_reward += reward

                godot = Godot(data)
                state = godot.get_state()

                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = actor(state_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = critic(state_tensor).squeeze()

                if prev_state is not None:
                    buffer.store(prev_state, prev_action, reward, state, done,
                                 log_prob=prev_log_prob.item(),
                                 value=prev_value.item(),
                                 next_value=value.item())

                await send_to_godot(websocket, action.item())
                prev_state, prev_value, prev_log_prob, prev_action = state, value, log_prob, action.item()

                if done == 1 or agent_die == 1 or steps == 2000:
                    break

            buffer.compute_gae(gamma=config.gamma, lam=config.lam)
            if buffer.size == 0:
                episode -= 1
                prev_state = None
                continue
            data_buffer_tensor = buffer.to_tensor()

            for _ in range(config.update_epochs):
                logits = actor(data_buffer_tensor["states"])

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(data_buffer_tensor["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - data_buffer_tensor["log_probs"])
                surr1 = ratio * data_buffer_tensor["advantages"]
                surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * data_buffer_tensor["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean() - config.entropy_coef * entropy

                value = critic(data_buffer_tensor["states"]).squeeze()
                critic_loss = F.mse_loss(value, data_buffer_tensor["returns"].squeeze())

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                critic_optimizer.step()

            print(f"🎯 Episode {episode} Reward: {episode_reward:.2f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
            save_model(actor, critic)

            rewards_per_episode.append(episode_reward)
            actor_loss_per_episode.append(actor_loss.item())
            critic_loss_per_episode.append(critic_loss.item())

            log_pot ={
                "rewards_per_episode": rewards_per_episode,
                "actor_loss_per_episode": actor_loss_per_episode,
                "critic_loss_per_episode": critic_loss_per_episode,
            }
            with open("ppo_metrics.json", "w") as f:
                json.dump(log_pot, f)
            buffer.clear()
            prev_state = None
            episode_reward = 0


    except websockets.exceptions.ConnectionClosed:
        print("⚠️ Connection closed by client")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        traceback.print_exc()


    # === Main Server Launcher ===
async def main():
    config = PPOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim, action_dim = 12, 8  # For Godot state [agent xyz + goal xyz]
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)
    buffer = ReplayBuffer(state_dim, max_size=config.buffer_size)

    load_model(actor, critic)

    async def handler(websocket):
        await echo(websocket, actor, critic, actor_optimizer, critic_optimizer, buffer, config)

    server = await websockets.serve(handler, "localhost", 8765)
    print("✅ WebSocket Server running at ws://localhost:8765")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())