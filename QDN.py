# ===============================
# Imports & Utility Functions
# ===============================
import asyncio
import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import websockets



# ===============================
# Plot reward graph for analysis
# ===============================
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

# ===============================
# Save & Load Models
# ===============================
def save_model(agent, path="model/model.pth", target_path="model/target_model.pth", epsilon_path="model/epsilon.txt"):
    torch.save(agent.q_value.state_dict(), path)
    torch.save(agent.target_q_value.state_dict(), target_path)
    with open(epsilon_path, "w") as f:
        f.write(str(agent.epsilon))

def load_model(agent, path="model/model.pth", target_path="model/target_model.pth", epsilon_path="model/epsilon.txt"):
    if os.path.exists(path):
        agent.q_value.load_state_dict(torch.load(path))
    if os.path.exists(target_path):
        agent.target_q_value.load_state_dict(torch.load(target_path))
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            agent.epsilon = float(f.read())

# ===============================
# Deep Q-Network
# ===============================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.0001):
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

# ===============================
# Agent Class for DQN Logic
# ===============================
class Agent:
    def __init__(self, learning_rate=0.001, gamma=0.85, epsilon=1.0, input_dim=13,
                 batch_size=128, action_dim=6, max_memory_size=100000,
                 epsilon_min=0.05, epsilon_decay=0.956):
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

        # Experience Replay Buffers
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
            print(q_values)
            return torch.argmax(q_values).item()


    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self.q_value.optimizer.zero_grad()
        batch_indices = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size, replace=False)

        # Prepare Batches
        state_batch = torch.tensor(self.state_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        action_batch = torch.tensor(self.action_memory[batch_indices], dtype=torch.int64).to(self.q_value.device)
        reward_batch = torch.tensor(self.reward_memory[batch_indices], dtype=torch.float32).to(self.q_value.device)
        done_batch = torch.tensor(self.done_memory[batch_indices], dtype=torch.bool).to(self.q_value.device)

        # Q-learning targets
        q_eval = self.q_value(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_q_value(next_state_batch)
            q_target = reward_batch + self.gamma * q_next.max(dim=1)[0] * (~done_batch)


        # Backpropagation
        loss = self.q_value.loss(q_eval, q_target)
        loss.backward()
        self.q_value.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % 100 == 0:
            self.target_q_value.load_state_dict(self.q_value.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ===============================
# Distance Helper Function
# ===============================
def distance_find(p1x, p1y, p1z, p2x, p2y, p2z):
    return math.sqrt((p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2)

def angle_diff_deg(a1, a2):
    return (a1 - a2 + 180) % 360 - 180
# ===============================
# WebSocket RL Agent Server
# ===============================
async def echo(websocket):
    agent = Agent()
    load_model(agent)

    episode = 0
    episode_end = 100
    a_diff =0
    step = 1

    total_reward = 0.0
    rewards_per_episode = []
    await websocket.send("start")

    try:
        while episode < episode_end:
            epoch_count = 0
            epoch_max = 5000


            while epoch_count < epoch_max:
                message = await websocket.recv()

                # Parse agent + environment state
                try:
                    status = json.loads(message)
                    skip = False
                except json.decoder.JSONDecodeError:
                    skip = True

                if not skip:
                    # Unpack data from environment
                    # sausage = [stats_Agent, stats_Target, vetor_x, vetor_z, state_Wall]
                    agent_status = status[0]
                    target_status = status[1]
                    walls = status[4]

                    # Boolean flags
                    hit_the_wall = walls[1]
                    saw_target = walls[2]
                    saw_walls = walls[3]

                    # Velocity
                    v_x = status[2]
                    v_z = status[3]

                    # Walls positions
                    # w1_x, w1_y, w1_z = walls[1], walls[2], walls[3]
                    # w2_x, w2_y, w2_z = walls[4], walls[5], walls[6]
                    # w3_x, w3_y, w3_z = walls[7], walls[8], walls[9]

                    # Agent and Target positions

                    a_x, a_y, a_z = agent_status[1], agent_status[2], agent_status[3]
                    r_y = agent_status[4]

                    t_x, t_y, t_z = target_status[1], target_status[2], target_status[3]

                    # Construct current state
                    state = [
                        a_x, a_y, a_z,  # ตำแหน่ง Agent
                        r_y,  # มุมมอง/การหมุนของ Agent
                        a_diff,  # ความต่างของมุมระหว่าง Agent กับ Target
                        t_x, t_y, t_z,  # ตำแหน่ง Target
                        v_x, v_z,  # ความเร็วของ Agent
                        hit_the_wall,  # ชนกำแพง
                        saw_target,  # เห็นเป้าหมาย
                        saw_walls  # เห็นกำแพง
                    ]

                    # Select action
                    action = agent.choose_action(state)

                    # Simulate next state (agent movement or rotation)
                    new_a_x, new_a_y, new_a_z = a_x, a_y, a_z
                    new_r_y = r_y

                    if action == 0:  # forward
                        radian_y = math.radians(r_y)
                        dir_x = math.cos(radian_y)
                        dir_z = math.sin(radian_y)
                        new_a_x += dir_x * step
                        new_a_z += dir_z * step

                    elif action == 1:  # backward
                        radian_y = math.radians(r_y)
                        dir_x = math.cos(radian_y)
                        dir_z = math.sin(radian_y)
                        new_a_x -= dir_x * step
                        new_a_z -= dir_z * step

                    elif action == 2:  # left (เดินขวางไปด้านซ้ายของ agent)
                        radian_y = math.radians(r_y + 90)
                        dir_x = math.cos(radian_y)
                        dir_z = math.sin(radian_y)
                        new_a_x += dir_x * step
                        new_a_z += dir_z * step

                    elif action == 3:  # right (เดินขวางไปด้านขวาของ agent)
                        radian_y = math.radians(r_y - 90)
                        dir_x = math.cos(radian_y)
                        dir_z = math.sin(radian_y)
                        new_a_x += dir_x * step
                        new_a_z += dir_z * step

                    if action == 4:  # rotate left
                        new_r_y += 4.77  # ปรับให้เท่ากับ Godot (สมมติ 60 FPS)
                    elif action == 5:  # rotate right
                        new_r_y -= 4.77

                    # Construct next state
                    next_state = [
                        new_a_x, new_a_y, new_a_z,
                        new_r_y,
                        a_diff,
                        t_x, t_y, t_z,
                        v_x, v_z,
                        hit_the_wall,
                        saw_target,
                        saw_walls
                    ]

                    # Calculate distances
                    distance = distance_find(new_a_x, new_a_y, new_a_z, t_x, t_y, t_z)
                    direction_to_target = math.atan2(t_z - a_z, t_x - a_x) * 180 / math.pi
                    angle_diff = angle_diff_deg(r_y, direction_to_target)
                    a_diff = angle_diff / 180.0

                    # === Reward Calculation ===
                    done = False
                    reward = 0.0
                    true_reward = 0.0

                    if distance > 29.0:
                        reward += -100.0
                        true_reward = -100.0

                    elif distance < 1.0:
                        reward += 100.0
                        true_reward = 100.0
                        done = True

                    if hit_the_wall:
                        reward += -2
                        true_reward += -1.0

                    if saw_target:
                        reward += 50 * (1 / (distance + 1))
                        true_reward += 1.0

                    if action in [4, 5]:
                        reward -= 0.1

                    if not saw_target:
                        reward += -0.5
                        true_reward += -0.5

                    if abs(angle_diff) < 10.0:
                        reward += (180 - abs(angle_diff)) / 180 * 2.0
                    else:
                        reward += -abs(angle_diff) / 180 * 2.0

                    # Store transition and learn
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.learn()

                    # Send command to environment
                    command = {
                        0: "forward", 1: "backward", 2: "left", 3: "right",
                        4: "rotate_left", 5: "rotate_right"
                    }.get(action, "stop")

                    total_reward += reward
                    epoch_count += 1

                    if epoch_count % 100 == 0:
                        save_model(agent)

                    await websocket.send(command)
                    print(
                        f"ep {episode}   Epoch:{epoch_count - 1} | Agent pos=({agent_status[1]:.2f}, {agent_status[2]:.2f}, {agent_status[3]:.2f}) "
                        f"Target=({target_status[1]:.2f}, {target_status[2]:.2f}, {target_status[3]:.2f}) "
                        f"| Reward={reward:.2f}"
                    )

                    # Episode End
                    if done or epoch_count % 500 == 0:
                        print(f"Episode {episode} Epoch:{epoch_count} reward:{total_reward}")
                        rewards_per_episode.append(total_reward)
                        with open("rewards_per_episode.json", "w") as f:
                            json.dump(rewards_per_episode, f)
                        save_model(agent)
                        episode += 1
                        break

            episode += 1
    except:
        print(f"❌ Unexpected server error")

# ===============================
# Start WebSocket Server
# ===============================
async def main():
    server = await websockets.serve(echo, "localhost", 8765)
    print("✅ Server started at ws://localhost:8765")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
