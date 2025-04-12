import asyncio
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import websockets
import numpy as np
import time
import math

from QDN import save_model


# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Match training model: 128 units
        self.fc2 = nn.Linear(128, 256)       # Match training model: 256 units
        self.fc3 = nn.Linear(256, action_dim) # Match training model: action_dim=4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Predict function
def predict_action(model, state):
    state_tensor = torch.tensor([state], dtype=torch.float32).to(model.device)
    with torch.no_grad():
        q_values = model(state_tensor)
    action = torch.argmax(q_values, dim=1).item()
    return action, q_values.cpu().numpy()

# WebSocket inference
async def echo(websocket):
    # Load trained model
    model = DQN(input_dim=8, action_dim=4)  # [a_x, a_y, a_z, t_x, t_y, t_z, v_x, v_z]
    model.load_state_dict(torch.load("model.pth", map_location=model.device))
    model.eval()
    skip = False

    episode = 0
    episode_end = 100

    total_reward = 0.0
    rewards_per_episode = []
    await websocket.send("start")

    while episode < episode_end:
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
                status = json.loads(message)

            # Extract data
            agent_status = status[0]
            target_status = status[1]
            v_x = float(status[4])
            v_z = float(status[5])
            a_x, a_y, a_z = float(agent_status[1]), float(agent_status[2]), float(agent_status[3])
            t_x, t_y, t_z = float(target_status[1]), float(target_status[2]), float(target_status[3])

            state = [a_x, a_y, a_z, t_x, t_y, t_z, v_x, v_z]
            new_a_x, new_a_y, new_a_z = a_x, a_y, a_z

            if action == 0: new_a_x += 1
            elif action == 1: new_a_x -= 1
            elif action == 2:new_a_z -= 1
            elif action == 3:new_a_z += 1

            distance = math.sqrt((new_a_x - t_x) ** 2 + (new_a_y - t_y) ** 2 + (new_a_z - t_z) ** 2)

            if distance > 30.0:
                true_reward = -100
                done = False
            elif distance < 2:
                true_reward = 100
                done = True
            else:
                true_reward = -10
                done = False

            # Predict action
            action, q_vals = predict_action(model, state)
            reaction = {0: "forward", 1: "backward", 2: "left", 3: "right"}.get(action, "stop")

            # Logging
            epoch_count += 1
            await websocket.send(reaction)

            # Handle episode end
            if done:
                print(f"Episode {episode} Epoch:{epoch_count} reward:{total_reward}")
                rewards_per_episode.append(total_reward)

                with open("rewards_per_episode.json", "w") as f:
                    json.dump(rewards_per_episode, f)
                episode += 1
                break


# Server setup
async def main():
    server = await websockets.serve(echo, "localhost", 8765)
    print("✅ Server started at ws://localhost:8765")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())