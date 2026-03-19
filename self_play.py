import numpy as np
from typing import List

class SelfPlayGame:
    """Single game for self-play"""
    
    def __init__(self, env, network, mcts_class, mcts_kwargs: dict, temperature: float = 1.0):
        self.env = env
        self.network = network
        self.mcts_class = mcts_class
        self.mcts_kwargs = mcts_kwargs
        self.temperature = temperature
    
    def play(self) -> List[tuple]:
        self.env.reset()
        game_history = []
        
        # 1. KHỞI TẠO MCTS ĐÚNG 1 LẦN (Để tận dụng Tree Reuse)
        mcts = self.mcts_class(self.env, self.network, **self.mcts_kwargs)
        move_count = 0

        while True: 
            board = self.env.board
            player = self.env.current_player

            # MCTS Search (Không khởi tạo lại mcts ở đây nữa)
            temp = 1.0 if move_count < 15 else 0.05
            action_probs, _ = mcts.search(board, player, temperature=temp)
            
            # Select action with temperature
            if self.temperature > 0:
                probs = np.power(action_probs + 1e-8, 1.0 / self.temperature)
                probs /= probs.sum() + 1e-8
                action = np.random.choice(len(probs), p=probs)
            else:
                action = np.argmax(action_probs)
            
            # Prepare tensor for buffer (Nên dùng .cpu() trước khi .numpy() để tránh lỗi nếu đang dùng GPU)
            state_tensor = self.network.prepare_input(board, player)
            state_tensor = state_tensor.squeeze(0).cpu().numpy()
            
            game_history.append({
                'state_tensor': state_tensor,
                'action_probs': action_probs.copy(),
                'player': player
            })
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                # 2. LẤY WINNER TỪ INFO
                winner = info.get("winner", getattr(self.env, "winner", 0))
                break
                
            # Cập nhật gốc cây xuống nút con tương ứng với action vừa đánh
            # (Chú ý: Đảm bảo trong class MCTS của bạn đã định nghĩa hàm update_root nhé)
            if hasattr(mcts, 'update_root'):
                mcts.update_root(action)
            move_count += 1
            
        episode_data = []
        
        # Gán nhãn outcome
        for experience in game_history:
            if winner == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if experience['player'] == winner else -1.0
                
            episode_data.append((
                experience['state_tensor'], 
                experience['action_probs'], 
                outcome
            ))
            
        return episode_data