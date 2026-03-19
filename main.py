import torch
import numpy as np
import os
from datetime import datetime


from vectorized_gomoku_env import VectorizedGomokuEnv
from batched_mcts import BatchedMCTS
from batched_self_play import BatchedSelfPlayGame

# Giả sử bạn lưu các class vào các thư mục tương ứng
from network import GomokuNet
from buffer import ReplayBuffer
from trainer import Trainer

class AlphaZeroGomoku:
    """Main AlphaZero training loop"""
    
    def __init__(self, 
                 num_iterations: int = 100,
                 num_games_per_iteration: int = 32,
                 num_mcts_simulations: int = 400,
                 batch_size: int = 128,
                 steps_per_iteration: int = 50,
                 replay_buffer_size: int = 300000,
                 num_parallel_games: int = 16):  
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.network = GomokuNet(board_size=15, num_residual_blocks=6, channels=64).to(self.device)
            
        self.trainer = Trainer(self.network, self.device, lr=0.001, l2_regularization=1e-4)
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Hyperparameters
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.num_mcts_simulations = num_mcts_simulations
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        self.num_parallel_games = num_parallel_games
        self.start_iteration = 0 # Biến đánh dấu vòng lặp bắt đầu
        
        # Checkpoint directory
        self.checkpoint_dir = '/kaggle/working/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, file_path: str):
        """Nạp trọng số Model và Optimizer để train tiếp"""
        if os.path.exists(file_path):
            print(f"=> Đang nạp checkpoint từ: {file_path}")
            checkpoint = torch.load(file_path, map_location=self.device)
            
            # Xử lý an toàn: Bóc lớp vỏ DataParallel nếu có
            raw_model = self.network.module if hasattr(self.network, 'module') else self.network
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            raw_model = raw_model.to(self.device)
            raw_model.eval()      
            

            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iteration = checkpoint['iteration'] + 1
            print(f"=> Nạp thành công! Sẽ huấn luyện tiếp từ vòng {self.start_iteration + 1}")
        else:
            print(f"❌ KHÔNG TÌM THẤY CHECKPOINT: {file_path}. Bắt đầu train từ đầu!")
    
    def train(self):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Target Iterations: {self.num_iterations}")
        print(f"Games per iteration: {self.num_games_per_iteration}")
        print(f"MCTS simulations: {self.num_mcts_simulations}\n")
        
        for iteration in range(self.start_iteration, self.num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Self-play (Parallel)
            print("\n[1/3] Running self-play games (Parallel)...")
            self.network.eval() 
            experiences = self._run_self_play()
            print(f"      Generated {len(experiences)} game states (before augmentation)")
            
            # Step 2: Add to replay buffer
            print("[2/3] Adding experiences to replay buffer with Augmentation...")
            for state_tensor, action_probs, outcome in experiences:
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
        """Run self-play games with BATCHED MCTS on GPU"""
        print(f"      Đang khởi chạy {self.num_games_per_iteration} ván cờ song song bằng Batched MCTS...")
        
        mcts_kwargs = {
            "num_simulations": self.num_mcts_simulations, 
            "c_puct": 1.5,
            "board_size": 15
        }
        
        # 1. Khởi tạo môi trường Vectorized (nhiều bàn cờ cùng lúc)
        vec_env = VectorizedGomokuEnv(num_envs=self.num_games_per_iteration, board_size=15, win_condition=5)
        
        # Lấy model raw để chạy
        raw_model = self.network.module if hasattr(self.network, 'module') else self.network
        raw_model.eval()
        # 2. Khởi tạo Batched Self Play Game
        game = BatchedSelfPlayGame(
            vec_env=vec_env, 
            network=raw_model, 
            batched_mcts_class=BatchedMCTS, 
            mcts_kwargs=mcts_kwargs, 
            num_target_games=self.num_games_per_iteration
        )
        
        try:
            with torch.no_grad():
                experiences = game.play()
            print(f"      Hoàn thành self-play! Thu thập được {len(experiences)} states.")
            return experiences
        except Exception as e:
            print(f"      Lỗi trong quá trình Batched Self-Play: {e}")
            return []
    
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
    # Configuration thực tế cho Kaggle
    config = {
        "num_iterations": 100,
        "num_games_per_iteration": 32,
        "num_mcts_simulations": 120,
        "batch_size": 256,
        "steps_per_iteration": 200,
        "replay_buffer_size": 80000,
        "num_parallel_games": 1
    }
    
    # Train
    agent = AlphaZeroGomoku(**config)
    
    # Đảm bảo đường dẫn này khớp với thư mục Input trên Kaggle của bạn
    checkpoint_path = "model_iter_0000.pt" 
    agent.load_checkpoint(checkpoint_path)
    
    agent.train()