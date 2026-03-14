import torch
import numpy as np
from GameEnv import GomokuEnv
from network import GomokuNet
from mcts import MCTS
from tqdm import tqdm


def load_ai(model_path, device):
    """Hàm tải siêu trí tuệ, tự động xử lý lỗi DataParallel nếu có"""
    model = GomokuNet(board_size=15, num_residual_blocks=10, channels=128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Bóc lớp vỏ DataParallel nếu file được save từ môi trường đa GPU (Kaggle)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang thiết lập Đấu Trường (Arena) trên {device}...\n")
    
    # ==========================================
    # 1. NẠP CÁC ĐẤU THỦ
    # ==========================================
    try:
        player1_model = load_ai("model_iter_0022.pt", device)
        player2_model = load_ai("model_iter_0012.pt", device)
        
    except Exception as e:
        print(f"❌ Lỗi khi tải Model: {e}\nHãy chắc chắn 2 file .pt đang nằm cùng thư mục.")
        return

    env = GomokuEnv(board_size=15, win_condition=5)
    NUM_GAMES = 10
    MCTS_SIMULATIONS = 400
    
    # Bảng điểm dùng tên làm key để tự động cộng đúng người
    score = {"Vòng 5": 0, "Vòng 12": 0, "Hòa": 0}
    
    print("="*50)
    print(f"🔥 GIẢI ĐẤU TỬ THẦN - {NUM_GAMES} VÁN 🔥")
    print("Luật: Đổi bên luân phiên sau mỗi ván")
    print("="*50 + "\n")

    # ==========================================
    # 2. VÒNG LẶP THI ĐẤU
    # ==========================================
    for game in tqdm(range(NUM_GAMES), desc="Tiến độ Giải đấu"):
        env.reset()
        
        # Đảo bên luân phiên cho công bằng
        if game % 2 == 0:
            black_name, white_name = "Vòng 5", "Vòng 12"
            # Lưu ý: Nếu class MCTS của bạn yêu cầu truyền 'env', hãy thêm env vào: MCTS(env, model...)
            mcts_p1 = MCTS(env, player1_model, num_simulations=MCTS_SIMULATIONS) 
            mcts_p2 = MCTS(env, player2_model, num_simulations=MCTS_SIMULATIONS)
        else:
            black_name, white_name = "Vòng 12", "Vòng 5"
            mcts_p1 = MCTS(env, player2_model, num_simulations=MCTS_SIMULATIONS) 
            mcts_p2 = MCTS(env, player1_model, num_simulations=MCTS_SIMULATIONS)
        
        steps = 0
        while True:
            current_player = env.current_player
            
            # Chọn não bộ tương ứng để suy nghĩ (p1 luôn là Đen/Đi trước, p2 luôn là Trắng/Đi sau)
            active_mcts = mcts_p1 if current_player == 1 else mcts_p2
            
            with torch.inference_mode():
                action_probs, _ = active_mcts.search(env.board.copy(), current_player)
            
            # Lựa chọn có chứa nhiễu (exploration) để tránh việc đánh đi đánh lại 1 ván
            action = np.argmax(action_probs)
            
            # Thực hiện nước đi
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Cập nhật nước đi vừa đánh cho CẢ HAI cây suy nghĩ
            mcts_p1.update_root(action)
            mcts_p2.update_root(action)
            
            if terminated or truncated:
                winner = info.get("winner", 0)
                if winner == 1:
                    score[black_name] += 1
                    tqdm.write(f"⚔️ Ván {game+1:2d} | ⬛ {black_name} THẮNG! (Sau {steps} nước)")
                elif winner == 2:
                    score[white_name] += 1
                    tqdm.write(f"⚔️ Ván {game+1:2d} | ⬜ {white_name} THẮNG! (Sau {steps} nước)")
                else:
                    score["Hòa"] += 1
                    tqdm.write(f"⚔️ Ván {game+1:2d} | 🤝 HÒA! (Sau {steps} nước)")
                break

    # ==========================================
    # 3. TỔNG KẾT BẢNG XẾP HẠNG
    # ==========================================
    print("\n" + "="*40)
    print("🏆 KẾT QUẢ CHUNG CUỘC 🏆")
    print("="*40)
    print(f"🦊 Vòng 5 thắng  : {score['Vòng 5']}")
    print(f"🐺 Vòng 12 thắng : {score['Vòng 12']}")
    print(f"🤝 Số ván hòa    : {score['Hòa']}")
    print("="*40)

if __name__ == "__main__":
    main()