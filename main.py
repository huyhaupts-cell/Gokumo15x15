import torch
import numpy as np
import os
from datetime import datetime

# Giả sử bạn lưu các class vào các thư mục tương ứng
from GameEnv import GomokuEnv
from network import GomokuNet
from mcts import MCTS
from buffer import ReplayBuffer
from self_play import SelfPlayGame # Không dùng SelfPlayParallel cho an toàn với GPU
from trainer import Trainer

class AlphaZeroGomoku:
    """Main AlphaZero training loop"""
    
    def __init__(self, 
                 num_iterations: int = 100,
                 num_games_per_iteration: int = 32,
                 num_mcts_simulations: int = 400, # Giảm xuống 400 để test nhanh hơn
                 batch_size: int = 128,
                 steps_per_iteration: int = 50):  # Đổi tên cho đúng với Trainer
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.env = GomokuEnv(board_size=15, win_condition=5)
        # Sửa lại params mạng theo class GomokuNet chuẩn: (board_size, num_residual_blocks, channels)
        self.network = GomokuNet(board_size=15, num_residual_blocks=10, channels=128).to(self.device)
        
        # BẬT TÍNH NĂNG CHẠY SONG SONG TRÊN NHIỀU GPU CỦA PYTORCH
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.network = torch.nn.DataParallel(self.network)
        self.trainer = Trainer(self.network, self.device, lr=0.001, l2_regularization=1e-4)
        self.replay_buffer = ReplayBuffer(capacity=500000)
        
        # Hyperparameters
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_mcts_simulations = num_mcts_simulations
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        
        # Checkpoint directory
        self.checkpoint_dir = '/kaggle/working/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Iterations: {self.num_iterations}")
        print(f"Games per iteration: {self.num_games_per_iteration}")
        print(f"MCTS simulations: {self.num_mcts_simulations}\n")
        
        for iteration in range(self.num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Self-play
            print("\n[1/3] Running self-play games...")
            self.network.eval() # Bật chế độ suy luận
            experiences = self._run_self_play()
            print(f"      Generated {len(experiences)} game states (before augmentation)")
            
            # Step 2: Add to replay buffer
            print("[2/3] Adding experiences to replay buffer with Augmentation...")
            for state_tensor, action_probs, outcome in experiences:
                # Add trực tiếp, ReplayBuffer đã có sẵn hàm get_symmetries để nhân 8 dữ liệu
                self.replay_buffer.add(state_tensor, action_probs, outcome)
            print(f"      Replay buffer size: {len(self.replay_buffer)}")
            
            # Step 3: Train network
            print(f"[3/3] Training network ({self.steps_per_iteration} steps)...")
            avg_policy_loss, avg_value_loss = self._train_network()
            print(f"      Policy loss: {avg_policy_loss:.4f}")
            print(f"      Value loss:  {avg_value_loss:.4f}")
            
            # Step 4: Save checkpoint
            self._save_checkpoint(iteration)
            print(f"      Checkpoint saved")
    
    def _run_self_play(self) -> list:
        """Run self-play games and collect experiences"""
        all_experiences = []
        
        mcts_kwargs = {
            "num_simulations": self.num_mcts_simulations, 
            "c_puct": 1.5
        }
        
        for game_id in range(self.num_games_per_iteration):
            # Cần tạo môi trường mới hoặc gọi env.reset() trong SelfPlayGame
            env = GomokuEnv(board_size=15, win_condition=5)
            
            game = SelfPlayGame(
                env,
                self.network,
                MCTS,
                mcts_kwargs,
                temperature=1.0
            )
            
            try:
                # Hàm play() trả về List[(state_tensor, action_probs, outcome)]
                with torch.no_grad():
                    experiences = game.play()
                all_experiences.extend(experiences)
                print(f"      Game {game_id+1}/{self.num_games_per_iteration} finished. Steps: {len(experiences)}")
            except Exception as e:
                print(f"      Error in game {game_id+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_experiences
    
    def _train_network(self) -> tuple:
        """Train network on replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            print("      Not enough data to train. Skipping...")
            return 0.0, 0.0
        
        return self.trainer.train_epoch(
            self.replay_buffer,
            batch_size=self.batch_size,
            steps=self.steps_per_iteration
        )
    
    def _save_checkpoint(self, iteration: int):

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"model_iter_{iteration+1:04d}.pt"
        )

        model = self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network

        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
        }, checkpoint_path)


if __name__ == '__main__':
    # Configuration
    config = {
        'num_iterations': 50,
        'num_games_per_iteration': 50, # Giảm xuống 10 lúc đầu để test luồng
        'num_mcts_simulations': 400,   # Giảm xuống 200 lúc đầu cho nhanh
        'batch_size': 512,
        'steps_per_iteration': 150
    }
    
    # Train
    agent = AlphaZeroGomoku(**config)
    agent.train()