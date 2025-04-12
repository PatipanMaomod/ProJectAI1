import asyncio
import json
import os
import math
from platform import win32_ver

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import websockets

def plot_rewards(rewards):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.savefig('reward_plot.png')
    plt.show()

def save_model(agent, path="/model/model.pth", target_path="/model/target_model.pth", epsilon_path="/model/epsilon.txt"):
    torch.save(agent.q_value.state_dict(), path)
    torch.save(agent.target_q_value.state_dict(), target_path)
    with open(epsilon_path, "w") as f:
        f.write(str(agent.epsilon))


def load_model(agent, path="/model/model.pth", target_path="/model/target_model.pth", epsilon_path="/model/epsilon.txt"):
    if os.path.exists(path):
        agent.q_value.load_state_dict(torch.load(path))
    if os.path.exists(target_path):
        agent.target_q_value.load_state_dict(torch.load(target_path))
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            agent.epsilon = float(f.read())


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)


        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, input_dim=18,
                 batch_size=128, action_dim=4, max_memory_size=10000,
                 epsilon_min=0.01, epsilon_decay=0.956):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.action_space = list(range(action_dim))
        self.memory_size = max_memory_size
        self.memory_counter = 0
        self.learn_step_counter = 0

        self.q_value = DQN(input_dim, action_dim, learning_rate)
        self.target_q_value = DQN(input_dim, action_dim, learning_rate)
        self.target_q_value.load_state_dict(self.q_value.state_dict())
        self.target_q_value.eval()

        self.state_memory = np.zeros((max_memory_size, input_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_memory_size, input_dim), dtype=np.float32)
        self.action_memory = np.zeros(max_memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(max_memory_size, dtype=np.float32)
        self.done_memory = np.zeros(max_memory_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.q_value.device)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        with torch.no_grad():
            q_values = self.q_value(state)
            #print(q_values)
            x = torch.argmax(q_values).item()

            return x

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self.q_value.optimizer.zero_grad()
        batch_indices = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        action_batch = torch.tensor(self.action_memory[batch_indices], dtype=torch.int64).to(self.q_value.device)
        reward_batch = torch.tensor(self.reward_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        done_batch = torch.tensor(self.done_memory[batch_indices], dtype=torch.bool).to(self.q_value.device)

        q_eval = self.q_value(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_q_value(next_state_batch)
            q_target = reward_batch + self.gamma * q_next.max(dim=1)[0] * (~done_batch)

        loss = self.q_value.loss(q_eval, q_target)
        loss.backward()
        self.q_value.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % 1000 == 0:
            self.target_q_value.load_state_dict(self.q_value.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def distance_find(p1x,p1y,p1z,p2x,p2y,p2z):
    d = math.sqrt((p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2)
    return d



async def echo(websocket):
    agent = Agent()
    load_model(agent)
    skip = False

    episode = 0
    episode_end = 100

    total_reward = 0.0
    rewards_per_episode = []
    await websocket.send("start")
    try:
        while episode<episode_end:
            epoch_count = 0
            epoch_max = 10000
            while epoch_count < epoch_max:
                message = await websocket.recv()
                try:
                    status = json.loads(message)
                    skip = False
                except json.decoder.JSONDecodeError:
                    status = None
                    skip = True
                if not skip:
                    agent_status = status[0]
                    target_status = status[1]
                    walls = status[6]
                    saw_walls = status[7]
                    v_x = status[4]
                    v_z = status[5]

                    w1_x,w1_y,w1_z = walls[1],walls[2],walls[3]
                    w2_x,w2_y,w2_z = walls[4],walls[5],walls[6]
                    w3_x,w3_y,w3_z = walls[7],walls[8],walls[9]

                    a_x, a_y, a_z = agent_status[1], agent_status[2], agent_status[3]
                    t_x, t_y, t_z = target_status[1], target_status[2], target_status[3]

                    state = [a_x, a_y, a_z, t_x, t_y, t_z, v_x, v_z, w1_x, w1_y, w1_z, w2_x, w2_y, w2_z, w3_x, w3_y, w3_z,saw_walls]

                    action = agent.choose_action(state)

                    new_a_x, new_a_y, new_a_z = a_x, a_y, a_z
                    if action == 0: new_a_x += 1
                    elif action == 1: new_a_x -= 1
                    elif action == 2: new_a_z -= 1
                    elif action == 3: new_a_z += 1

                    next_state = [new_a_x, new_a_y, new_a_z, t_x, t_y, t_z, v_x, v_z, w1_x, w1_y, w1_z, w2_x, w2_y, w2_z, w3_x, w3_y, w3_z,saw_walls]
                    distance = distance_find(p1x=new_a_x, p1y=new_a_y, p1z=new_a_z, p2x=t_x, p2y=t_y, p2z=t_z)

                    distance_w1 = distance_find(p1x=w1_x, p1y=w1_y, p1z=w1_z, p2x=new_a_x, p2y=new_a_y, p2z=new_a_z)
                    distance_w2 = distance_find(p1x=w2_x, p1y=w2_y, p1z=w2_z, p2x=new_a_x, p2y=new_a_y, p2z=new_a_z)
                    distance_w3 = distance_find(p1x=w3_x, p1y=w3_y, p1z=w3_z, p2x=new_a_x, p2y=new_a_y, p2z=new_a_z)

                    if distance > 30.0:
                        reward = -100.0
                        true_reward = -100
                        done = False
                        print("Fall")
                    elif distance < 2:
                        reward = 100.0
                        true_reward = 100
                        done = True

                    elif min(distance_w1, distance_w2, distance_w3) < 1 or saw_walls:
                        reward = -0.1 * epoch_count - distance
                        print('ชน',saw_walls)
                        true_reward = -1
                        done = False

                    else:
                        reward = -distance
                        true_reward = -1
                        done = False

                    agent.store_transition(state, action, reward, next_state, done)
                    agent.learn()


                    command = {0: "forward", 1: "backward", 2: "left", 3: "right"}.get(action, "stop")
                    # print(f"Epoch {epoch_count}: Action={command}, Reward={reward:.2f}, Epsilon={agent.epsilon:.4f}")
                    # print(f"Agent: [{a_x:.2f}, {a_y:.2f}, {a_z:.2f}], Target: [{t_x:.2f}, {t_y:.2f}, {t_z:.2f}]")

                    total_reward += true_reward
                    epoch_count += 1

                    if epoch_count % 100 == 0:
                        save_model(agent)

                    await websocket.send(command)


                    if done:
                        print(f"Episode {episode} Epoch:{epoch_count} reward:{total_reward}")
                        rewards_per_episode.append(total_reward)
                        with open("rewards_per_episode.json", "w") as f:
                            json.dump(rewards_per_episode, f)
                        save_model(agent)
                        episode += 1
                        break
    except:
        print('End')
async def main():
    server = await websockets.serve(echo, "localhost", 8765)
    print("✅ Server started at ws://localhost:8765")
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except:
        print('')
