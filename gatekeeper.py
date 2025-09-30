# gatekeeper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class GatekeeperNet(nn.Module):
    def __init__(self, s_dim:int, a_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim)
        )
    def forward(self, x):
        return self.net(x)

class Gatekeeper:
    def __init__(self, state_dim:int, action_dim:int, device="cpu", lr=1e-4):
        self.device = device
        self.net = GatekeeperNet(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = []
        self.max_replay = 50000

    def select(self, state:np.ndarray, eps:float=0.1):
        if random.random() < eps:
            return random.randrange(self.net.net[-1].out_features)
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(s)
        return int(q.argmax().item())

    def store(self, s,a,r,s2,d):
        if len(self.replay) > self.max_replay:
            self.replay.pop(0)
        self.replay.append((s,a,r,s2,d))

    def train_step(self, batch_size=64, gamma=0.99):
        if len(self.replay) < batch_size:
            return 0.0
        batch = random.sample(self.replay, batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(self.device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        s2 = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32).to(self.device)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)
        q = self.net(s)
        q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.net(s2).max(1)[0]
            target = r + gamma * q2 * (1 - d)
        loss = F.mse_loss(q_a, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()
