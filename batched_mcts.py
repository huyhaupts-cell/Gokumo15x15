import math
import numpy as np
import torch
from mcts import MCTSNode # Dùng lại node của bạn

class BatchedMCTS:
    def __init__(self, network, num_envs: int, num_simulations=150, c_puct=1.5, board_size=15):
        self.network = network
        self.num_envs = num_envs
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.board_size = board_size
        
        # Một mảng chứa num_envs cái gốc (root)
        self.roots = [None] * num_envs
        # Không dùng cache ở đây vì gom batch thay đổi liên tục

    def get_candidate_moves(self, board):
        # TÌM CÁC Ô TRỐNG TRONG MỘT VÙNG CỐ ĐỊNH (ACTIVE ZONE)
        # Giới hạn AI chỉ được đánh trong vùng 9x9 ở trung tâm bàn cờ
        ACTIVE_SIZE = 9
        
        # Tính toán tọa độ giới hạn
        center = self.board_size // 2
        offset = ACTIVE_SIZE // 2
        min_x, max_x = max(0, center - offset), min(self.board_size, center + offset + 1)
        min_y, max_y = max(0, center - offset), min(self.board_size, center + offset + 1)
        
        moves = []
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if board[x, y] == 0:
                    moves.append(x * self.board_size + y)
                    
        return np.array(moves)

    def check_winner_fast(self, board, last_move):
        if last_move is None: return 0
        size = board.shape[0]
        x, y = divmod(last_move, size)
        player = board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            for step in range(1, 5):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player: count += 1
                else: break
            for step in range(1, 5):
                nx, ny = x - dx * step, y - dy * step
                if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player: count += 1
                else: break
            if count >= 5: return player
        return 0

    def update_root(self, env_idx, action):
        root = self.roots[env_idx]
        if root and action in root.children:
            self.roots[env_idx] = root.children[action]
            self.roots[env_idx].parent = None
        else:
            self.roots[env_idx] = None

    def search(self, boards_list, players_list, temperatures):
        device = next(self.network.parameters()).device
        B = self.num_envs

        # 1. Khởi tạo/Chuẩn bị Gốc
        roots_to_eval = []
        eval_boards = []
        eval_players = []

        for i in range(B):
            if self.roots[i] is None:
                self.roots[i] = MCTSNode(boards_list[i], player=players_list[i])
            if len(self.roots[i].children) == 0:
                roots_to_eval.append(i)
                eval_boards.append(boards_list[i])
                eval_players.append(players_list[i])

        # Batch Evaluate cho các Root trống
        if roots_to_eval:
            tensor_list = [self.network.prepare_input(b, p) for b, p in zip(eval_boards, eval_players)]
            batch_tensor = torch.cat(tensor_list, dim=0).to(device)
            
            with torch.no_grad():
                p_batch, v_batch = self.network(batch_tensor)
            
            p_batch = p_batch.cpu().numpy()
            v_batch = v_batch.cpu().numpy()

            for idx, env_i in enumerate(roots_to_eval):
                valid_moves = self.get_candidate_moves(eval_boards[idx])
                policy = p_batch[idx]
                mask = np.ones_like(policy) * float('-inf')
                mask[valid_moves] = 0
                
                logits = policy + mask
                logits = logits - np.max(logits)
                exp_p = np.exp(logits)
                p_prior = exp_p / (np.sum(exp_p) + 1e-8)
                
                self.roots[env_i].expand(valid_moves, p_prior, eval_players[idx])
                self.roots[env_i].value_sum += float(v_batch[idx][0])
                self.roots[env_i].visits += 1

        # 2. Thêm Dirichlet Noise
        for i in range(B):
            valid_moves = list(self.roots[i].children.keys())
            if len(valid_moves) > 0:
                epsilon = 0.25 if self.roots[i].visits < 30 else 0.05
                alpha = 0.03 * (self.board_size**2 / len(valid_moves))
                noise = np.random.dirichlet([alpha] * len(valid_moves))
                for j, move in enumerate(valid_moves):
                    self.roots[i].children[move].policy_prior = (1 - epsilon) * self.roots[i].children[move].policy_prior + epsilon * noise[j]

        # 3. Chạy Simulation đồng bộ
        for _ in range(self.num_simulations):
            leaf_nodes = []
            eval_boards = []
            eval_players = []
            eval_indices = []

            for i in range(B):
                node = self.roots[i]
                current_player = players_list[i]
                last_action = None

                # Selection
                while node.children:
                    action, node = node.select_child(self.c_puct)
                    current_player = 3 - current_player
                    last_action = action

                winner = self.check_winner_fast(node.board, last_action)
                if winner != 0:
                    node.backup(-1.0)
                    continue

                valid_moves = self.get_candidate_moves(node.board)
                if len(valid_moves) == 0:
                    node.backup(0.0)
                    continue

                # Node cần neural network
                leaf_nodes.append(node)
                eval_boards.append(node.board)
                eval_players.append(current_player)
                eval_indices.append(i)

            # Batch Evaluation Nút Cổ Chai
            if leaf_nodes:
                tensor_list = [self.network.prepare_input(b, p) for b, p in zip(eval_boards, eval_players)]
                batch_tensor = torch.cat(tensor_list, dim=0).to(device)
                
                with torch.no_grad():
                    p_batch, v_batch = self.network(batch_tensor)
                
                p_batch = p_batch.cpu().numpy()
                v_batch = v_batch.cpu().numpy()

                for idx, node in enumerate(leaf_nodes):
                    valid_moves = self.get_candidate_moves(node.board)
                    policy = p_batch[idx]
                    
                    mask = np.ones_like(policy) * float('-inf')
                    mask[valid_moves] = 0
                    logits = policy + mask
                    logits = logits - np.max(logits)
                    exp_p = np.exp(logits)
                    p_prior = exp_p / (np.sum(exp_p) + 1e-8)
                                        
                    val = max(-1.0, min(1.0, float(v_batch[idx][0])))
                    node.expand(valid_moves, p_prior, eval_players[idx])
                    node.backup(-val)

        # 4. Tính toán kết quả
        actions_probs_batch = []
        for i in range(B):
            action_probs = np.zeros(self.board_size ** 2)
            for move, child in self.roots[i].children.items():
                action_probs[move] = child.visits
                
            temp = temperatures[i]
            if temp == 0:
                best = np.argmax(action_probs)
                probs = np.zeros_like(action_probs)
                probs[best] = 1.0
            else:
                probs = np.power(action_probs, 1.0 / temp)
                probs /= np.sum(probs) + 1e-8
            actions_probs_batch.append(probs)

        return np.array(actions_probs_batch)