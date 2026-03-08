import numpy as np
from collections import deque
import random
from typing import Tuple

class ReplayBuffer:
    """Replay buffer for storing self-play experience"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def get_symmetries(self, state: np.ndarray, pi: np.ndarray) -> list:
        """Nhân bản trạng thái thành 8 góc độ khác nhau"""
        # state có shape là (2, 15, 15)
        # pi là mảng 1D (225,)
        board_size = state.shape[1]
        pi_board = np.reshape(pi, (board_size, board_size))
        symmetries = []
        
        for i in range(4):
            # Xoay i * 90 độ (Xoay hai kênh của state cùng lúc trên trục H và W)
            rot_state = np.rot90(state, k=i, axes=(1, 2))
            rot_pi = np.rot90(pi_board, k=i)
            symmetries.append((rot_state.copy(), rot_pi.ravel().copy()))
            
            # Lật ngang (Flip left-right)
            flip_state = np.flip(rot_state, axis=2)
            flip_pi = np.fliplr(rot_pi)
            symmetries.append((flip_state.copy(), flip_pi.ravel().copy()))
            
        return symmetries

    def add(self, state_tensor: np.ndarray, action_probs: np.ndarray, game_outcome: float):
        """
        Add experience to buffer with Data Augmentation
        
        Args:
            state_tensor: Tensor góc nhìn của AI (2, board_size, board_size)
            action_probs: MCTS action probabilities (board_size^2,)
            game_outcome: Final game outcome (+1 (thắng), 0 (hòa), -1 (thua))
        """
        # Áp dụng nhân 8 dữ liệu
        symm_data = self.get_symmetries(state_tensor, action_probs)
        
        for sym_state, sym_pi in symm_data:
            self.buffer.append((
                sym_state.astype(np.int8),
                sym_pi.astype(np.float16),
                np.float32(game_outcome)
            ))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch from buffer ready for PyTorch"""
        size = min(batch_size, len(self.buffer))
        idx = np.random.randint(0, len(self.buffer), size=size)
        batch = [self.buffer[i] for i in idx]
        
        size = len(batch)
        board_size = batch[0][0].shape[1]

        states = np.empty((size, 2, board_size, board_size), dtype=np.float32)
        pis = np.empty((size, board_size*board_size), dtype=np.float32)
        outcomes = np.empty(size, dtype=np.float32)

        for i, (s, p, o) in enumerate(batch):
            states[i] = s
            pis[i] = p
            outcomes[i] = o
        
        return states, pis, outcomes
    
    def __len__(self) -> int:
        return len(self.buffer)