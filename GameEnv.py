import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class GomokuEnv(gym.Env):
    """
    Gomoku (5 in a row) environment with 15x15 board
    Gym-compatible for easy training
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, board_size: int = 15, win_condition: int = 5):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_condition = win_condition
        
        # Action space: 225 possible positions (15x15)
        self.action_space = spaces.Discrete(self.board_size ** 2)
        
        # Observation space: 15x15 board with values 0 (empty), 1 (player), 2 (opponent)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2,self.board_size, self.board_size), 
            dtype=np.float32
        )
        
        self.reset()

    def _get_obs(self):
        player = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == (3 - self.current_player)).astype(np.float32)
        return np.stack([player, opponent])
    
    def _get_info(self) -> Dict[str, Any]:
        """Trả về action_mask để thuật toán RL biết ô nào bị cấm."""
        return {"action_mask": (self.board.ravel() == 0).astype(np.int8)}

    def reset(self, seed=None, options=None) -> np.ndarray:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.move_count = 0
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of environment dynamics
        
        Args:
            action: Position to place stone (0-224 for 15x15)
        
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_obs(), 0, True, False, self._get_info()
        
        x, y = divmod(action, self.board_size)
        
        # Check valid move
        if self.board[x, y] != 0:
            info = self._get_info()
            info["invalid_move"] = True
            return self._get_obs(), -10, True, False, info
        
        # Place stone
        self.board[x, y] = self.current_player
        self.move_count += 1
        
        # Check win
        winner = self._check_winner(x, y)
        
        if winner != 0:
            reward = 1.0 if winner == self.current_player else -1.0
            self.done = True
            info = self._get_info()
            info["winner"] = winner
            return self._get_obs(), reward, True, False, info
        
        # Check draw (board full)
        if self.move_count >= self.board_size ** 2:
            self.done = True
            info = self._get_info()
            info["draw"] = True
            return self._get_obs(), 0, True, False, info
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        return self._get_obs(), 0, False, False, self._get_info()
    
    def _check_winner(self, x, y):

        player = self.board[x, y]

        directions = [(1,0),(0,1),(1,1),(1,-1)]

        for dx,dy in directions:

            count = 1

            for step in range(1, self.win_condition):
                nx = x + dx*step
                ny = y + dy*step
                if 0<=nx<self.board_size and 0<=ny<self.board_size and self.board[nx,ny]==player:
                    count +=1
                else:
                    break

            for step in range(1, self.win_condition):
                nx = x - dx*step
                ny = y - dy*step
                if 0<=nx<self.board_size and 0<=ny<self.board_size and self.board[nx,ny]==player:
                    count +=1
                else:
                    break

            if count >= self.win_condition:
                return player

        return 0
    
    def get_valid_moves(self) -> np.ndarray:
        """Get all valid moves (empty positions)"""
        return np.where(self.board.ravel() == 0)[0]
    
    def render(self, mode: str = 'human'):
        """Render the board"""
        print("\n   ", end="")
        for i in range(self.board_size):
            print(f"{i:2}", end=" ")
        print()
        
        for i in range(self.board_size):
            print(f"{i:2} ", end="")
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    print(" .", end=" ")
                elif self.board[i, j] == 1:
                    print(" X", end=" ")
                else:
                    print(" O", end=" ")
            print()
        print()
    
    def close(self):
        pass