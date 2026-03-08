import math
import numpy as np
import torch
from typing import Dict, Tuple

class MCTSNode:
    """Nút trong cây MCTS"""
    def __init__(self, board_state: np.ndarray, parent=None, action: int = None, player: int = 1):
        self.board = board_state.copy() # SỬA LỖI 1: Thêm .copy()
        
        if action is not None and parent is not None:
            board_size = self.board.shape[0]
            x, y = divmod(action, board_size)
            self.board[x, y] = player
            
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.policy_prior = 0.0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct: float = 1.25) -> float: 
        exploration = c_puct * self.policy_prior * math.sqrt(self.parent.visits + 1) / (1 + self.visits)
        return self.q_value + exploration

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct) -> Tuple[int, 'MCTSNode']:
        best_action, best_child = max(
            self.children.items(),
            key=lambda item: item[1].ucb_score(c_puct)
        )
        return best_action, best_child

    def expand(self, valid_moves: np.ndarray, policy_priors: np.ndarray, next_player: int):
        for move in valid_moves:
            if move not in self.children:
                self.children[move] = MCTSNode(self.board, parent=self, action=move, player=next_player)
                self.children[move].policy_prior = policy_priors[move]

    def backup(self, value: float):
        self.visits += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backup(-value)


class MCTS:
    """Tìm kiếm Cây Monte Carlo với sự dẫn dắt của Mạng Nơ-ron"""
    
    def __init__(self, env, network, num_simulations: int = 800, c_puct: float = 1.25):
        self.env = env
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None

    def get_candidate_moves(self, board, radius=2):
        size = board.shape[0]
        moves = set()
        stones = np.argwhere(board != 0)

        if len(stones) == 0:
            return np.array([size*size//2])

        for x, y in stones:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if board[nx,ny] == 0:
                            moves.add(nx*size+ny)

        return np.array(list(moves))
        
    def update_root(self, action: int):
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    def search(self, board: np.ndarray, player: int) -> Tuple[np.ndarray, float]:
            if self.root is None:
                self.root = MCTSNode(board, player=player)
                
            root = self.root
            valid_moves = self.get_candidate_moves(board)
            device = next(self.network.parameters()).device

            # 1. Expand root nếu chưa được expand
            if not root.is_expanded() and len(valid_moves) > 0:
                input_tensor = self.network.prepare_input(board, player).to(device)
                
                with torch.no_grad():
                    policy, value = self.network(input_tensor)

                policy_logits = policy[0].clone()
                mask = torch.ones_like(policy_logits) * float('-inf')
                mask[valid_moves] = 0.0
                policy_priors = torch.softmax(policy_logits + mask, dim=0).cpu().numpy()

                root.expand(valid_moves, policy_priors, player)
                
                # SỬA LỖI 3: Khôi phục lại dòng backup cho root
                root.backup(value[0, 0].item())

            # 2. SỬA LỖI 2: Luôn bơm nhiễu ở đầu lượt search để đảm bảo Exploration
            if len(valid_moves) > 0:
                epsilon = 0.25
                alpha = 0.03
                noise = np.random.dirichlet([alpha] * len(valid_moves))
                
                sum_priors = 0.0
                for i, move in enumerate(valid_moves):
                    if move in root.children:
                        root.children[move].policy_prior = (1 - epsilon) * root.children[move].policy_prior + epsilon * noise[i]
                        sum_priors += root.children[move].policy_prior
                
                if sum_priors > 0:
                    for move in valid_moves:
                        if move in root.children:
                            root.children[move].policy_prior /= sum_priors

            # 3. Chạy Simulations
            for _ in range(self.num_simulations):
                node = root
                current_player = player
                
                while node.is_expanded():
                    action, node = node.select_child(self.c_puct)
                    current_player = 3 - current_player

                valid_moves = self.get_candidate_moves(node.board)
                
                # SỬA LỖI 1: Bỏ root.visits == 0 đi. Ở đây ta đang xét 'node' dưới đáy cây!
                if len(valid_moves) > 0:
                    input_tensor = self.network.prepare_input(node.board, current_player).to(device)
                    
                    with torch.no_grad():
                        policy, value = self.network(input_tensor)
                    
                    policy_logits = policy[0].clone()
                    mask = torch.ones_like(policy_logits) * float('-inf')
                    mask[valid_moves] = 0.0
                    policy_priors = torch.softmax(policy_logits + mask, dim=0).cpu().numpy()
                    value = value[0, 0].item()
                    
                    node.expand(valid_moves, policy_priors, current_player)
                else:
                    value = 0.0

                node.backup(-value)

            # 4. Trả kết quả
            action_probs = np.zeros(self.env.board_size ** 2)
            for move, child in root.children.items():
                action_probs[move] = child.visits
                
            total = np.sum(action_probs)
            if total > 0:
                action_probs /= total
            else:
                action_probs = np.ones_like(action_probs) / len(action_probs)
            
            return action_probs, -root.q_value