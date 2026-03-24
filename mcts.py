import math
import numpy as np
import torch


class MCTSNode:
    def __init__(self, board, parent=None, action=None, player=1):
        self.board = board.copy()

        if action is not None and parent is not None:
            size = self.board.shape[0]
            x, y = divmod(action, size)
            self.board[x, y] = player

        self.parent = parent
        self.action = action
        self.player = player

        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.policy_prior = 0.0

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct):
        prior = self.policy_prior
        return self.q_value + c_puct * prior * math.sqrt(self.parent.visits + 1) / (1 + self.visits)

    def select_child(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].ucb_score(c_puct))

    def expand(self, valid_moves, policy_priors, next_player):

        for move in valid_moves:
            if move not in self.children:
                child = MCTSNode(self.board, parent=self, action=move, player=next_player)
                child.policy_prior = policy_priors[move]
                self.children[move] = child

    def backup(self, value):
        value = max(-1.0, min(1.0, value))

        self.visits += 1
        self.value_sum += value

        if self.parent is not None:
            self.parent.backup(-value)


class MCTS:
    def __init__(self, env, network, num_simulations=400, c_puct=1.5):
        self.env = env
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.cache = {}


    def get_candidate_moves(self, board):
        size = board.shape[0]
        stones = np.argwhere(board != 0)

        if len(stones) < 10:
            return np.where(board.ravel() == 0)[0]

        moves = set()
        for x, y in stones:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if board[nx, ny] == 0:
                            moves.add(nx * size + ny)

        return np.array(list(moves))

    def update_root(self, action):
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    def get_action_probs(self, root, temperature=1.0):
        action_probs = np.zeros(self.env.board_size ** 2)

        for move, child in root.children.items():
            action_probs[move] = child.visits

        if temperature == 0:
            best = np.argmax(action_probs)
            probs = np.zeros_like(action_probs)
            probs[best] = 1.0
            return probs

        action_probs = action_probs ** (1.0 / temperature)
        action_probs /= np.sum(action_probs) + 1e-8

        return action_probs

    def check_winner_fast(self, board, last_move):
        if last_move is None:
            return 0

        size = board.shape[0]
        x, y = divmod(last_move, size)
        player = board[x, y]

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1

            for step in range(1, 5):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                    count += 1
                else:
                    break

            for step in range(1, 5):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return player

        return 0

    def search(self, board, player, temperature=1.0):
        if len(self.cache) > 20000:
            self.cache.clear()
        device = next(self.network.parameters()).device

        if self.root is None:
            self.root = MCTSNode(board, player=player)

        root = self.root

        valid_moves = self.get_candidate_moves(board)

        # Expand root
        if len(root.children) == 0:
            input_tensor = self.network.prepare_input(board, player).to(device)
            key = (board.tobytes(), player)

            if key in self.cache:
                policy, value = self.cache[key]
            else:
                with torch.no_grad():
                    policy, value = self.network(input_tensor)
                self.cache[key] = (policy, value)

            policy_logits = policy[0].clone()

            mask = torch.ones_like(policy_logits) * float('-inf')
            mask[valid_moves] = 0

            policy_priors = torch.softmax(policy_logits + mask, dim=0).cpu().numpy()

            root.expand(valid_moves, policy_priors, player)
            root.value_sum += value.item()
            root.visits += 1

        # Dirichlet noise (root only)
        if len(valid_moves) > 0:
            epsilon = 0.25 if root.visits < 30 else 0.05
            alpha = 0.03 * (self.env.board_size**2 / len(valid_moves))

            noise = np.random.dirichlet([alpha] * len(valid_moves))

            noise_dict = {move: noise[i] for i, move in enumerate(valid_moves)}

            for move, child in root.children.items():
                if move in noise_dict:
                    child.policy_prior = (1 - epsilon) * child.policy_prior + epsilon * noise_dict[move]

        # Simulations
        for _ in range(self.num_simulations):
            node = root
            current_player = player
            last_action = None

            # Selection
            while node.children:
                action, node = node.select_child(self.c_puct)
                current_player = 3 - current_player
                last_action = action

            # Terminal check
            winner = self.check_winner_fast(node.board, last_action)
            
            if winner != 0:
                value = -1.0
                node.backup(-value)
                continue

            valid_moves = self.get_candidate_moves(node.board)

            if len(valid_moves) == 0:
                node.backup(0.0)
                continue

            # NN evaluation
            input_tensor = self.network.prepare_input(node.board, current_player).to(device)
            key = (node.board.tobytes(), current_player)

            if key in self.cache:
                policy, value = self.cache[key]
            else:
                with torch.no_grad():
                    policy, value = self.network(input_tensor.to(device))
                self.cache[key] = (policy, value)

            policy_logits = policy[0].clone()

            mask = torch.ones_like(policy_logits) * float('-inf')
            mask[valid_moves] = 0

            policy_priors = torch.softmax(policy_logits + mask, dim=0).cpu().numpy()
            value = max(-1.0, min(1.0, value.item()))

            node.expand(valid_moves, policy_priors, current_player)
            node.backup(-value)

        return self.get_action_probs(root, temperature), root.q_value
