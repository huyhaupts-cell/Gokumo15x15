import numpy as np
import torch

class BatchedSelfPlayGame:
    def __init__(self, vec_env, network, batched_mcts_class, mcts_kwargs: dict, num_target_games: int):
        self.vec_env = vec_env
        self.network = network
        self.num_envs = vec_env.num_envs
        self.num_target_games = num_target_games
        self.mcts = batched_mcts_class(network, self.num_envs, **mcts_kwargs)
        
    def play(self) -> list:
        self.vec_env.reset()
        completed_games_data = []
        games_completed = 0
        
        game_histories = [[] for _ in range(self.num_envs)]
        move_counts = [0] * self.num_envs
        
        while games_completed < self.num_target_games:
            # Lấy board trực tiếp vì VecEnv đã cập nhật
            boards = [env.board.copy() for env in self.vec_env.envs]
            players = [env.current_player for env in self.vec_env.envs]
            temperatures = [1.0 if count < 15 else 0.05 for count in move_counts]
            
            # GỌI BATCHED MCTS (GPU TÍNH TOÁN CÙNG LÚC num_envs BÀN CỜ)
            with torch.no_grad():
                action_probs_batch = self.mcts.search(boards, players, temperatures)
            
            actions = []
            for i in range(self.num_envs):
                probs = action_probs_batch[i]
                if temperatures[i] > 0:
                    probs = probs / (probs.sum() + 1e-8)
                    action = np.random.choice(len(probs), p=probs)
                else:
                    action = np.argmax(probs)
                actions.append(action)
                
                state_tensor = self.network.prepare_input(boards[i], players[i]).squeeze(0).cpu().numpy()
                game_histories[i].append({
                    'state_tensor': state_tensor,
                    'action_probs': probs.copy(),
                    'player': players[i]
                })
                
            # Đẩy action vào VecEnv
            obs, rewards, dones, truncs, infos = self.vec_env.step(actions)
            
            for i in range(self.num_envs):
                if dones[i] or truncs[i]:
                    final_info = infos[i].get("final_info", infos[i])
                    winner = final_info.get("winner", 0)
                    
                    for exp in game_histories[i]:
                        outcome = 0.0 if winner == 0 else (1.0 if exp['player'] == winner else -1.0)
                        completed_games_data.append((exp['state_tensor'], exp['action_probs'], outcome))
                        
                    game_histories[i] = [] 
                    move_counts[i] = 0
                    self.mcts.roots[i] = None
                    games_completed += 1
                else:
                    move_counts[i] += 1
                    self.mcts.update_root(i, actions[i])
                    
            if games_completed > 0 and games_completed % 4 == 0:
                print(f"      [Batched Progress] Gom được {games_completed}/{self.num_target_games} ván cờ...")
                
        return completed_games_data