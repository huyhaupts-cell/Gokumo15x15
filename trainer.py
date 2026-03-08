import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple

class Trainer:
    """Network trainer cho AlphaZero"""
    
    def __init__(self, network, device, lr=0.001, l2_regularization=1e-4):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=l2_regularization)
        self.value_loss_fn = nn.MSELoss()
    
    def train_step(self, states: np.ndarray, action_probs: np.ndarray, 
                   outcomes: np.ndarray) -> Tuple[float, float]:

        self.network.train()

        states_tensor = torch.from_numpy(states).float().to(self.device)
        target_pis = torch.FloatTensor(action_probs).to(self.device)
        target_values = torch.FloatTensor(outcomes).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()

        policy_logits, value_pred = self.network(states_tensor)
        
        # Đã xóa dòng value_pred = torch.tanh(value_pred) ở đây vì mạng đã tự xuất tanh()

        # 1. Policy loss (Cross-Entropy)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_pis * log_probs).sum(dim=1).mean()

        # 2. Entropy bonus (Giúp chống hội tụ sớm)
        probs = torch.softmax(policy_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # 3. Value loss (MSE)
        value_loss = self.value_loss_fn(value_pred, target_values)

        # 4. Total loss
        # Tổng loss = Policy + Value - 0.01 * Entropy
        total_loss = policy_loss + value_loss - 0.01 * entropy

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

        self.optimizer.step()

        return policy_loss.item(), value_loss.item()
        
    def train_epoch(self, replay_buffer, batch_size: int = 128, steps: int = 50) -> Tuple[float, float]:
        """
        Train cho một số lượng steps nhất định trong Epoch.
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0
            
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(steps):
            # Lấy mẫu trực tiếp và đẩy vào train_step
            states, action_probs, outcomes = replay_buffer.sample(batch_size)
            
            p_loss, v_loss = self.train_step(states, action_probs, outcomes)
            
            total_policy_loss += p_loss
            total_value_loss += v_loss
            
        avg_policy_loss = total_policy_loss / steps
        avg_value_loss = total_value_loss / steps
        
        return avg_policy_loss, avg_value_loss