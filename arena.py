import pygame
import sys
import torch
import numpy as np
import time
from GameEnv import GomokuEnv
from network import GomokuNet
from mcts import MCTS

# ==========================================
# CẤU HÌNH GIAO DIỆN
# ==========================================
BOARD_SIZE = 15
GRID_SIZE = 40  
MARGIN = 40     
WINDOW_SIZE = BOARD_SIZE * GRID_SIZE + MARGIN * 2

COLOR_BG = (222, 184, 135)   
COLOR_LINE = (0, 0, 0)       
COLOR_BLACK = (20, 20, 20)   
COLOR_WHITE = (235, 235, 235)
COLOR_HIGHLIGHT = (255, 0, 0) # Màu đánh dấu nước đi vừa đánh

def draw_board(screen, board, last_action=None):
    screen.fill(COLOR_BG)
    
    # Vẽ lưới
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, COLOR_LINE, (MARGIN + i * GRID_SIZE, MARGIN), (MARGIN + i * GRID_SIZE, WINDOW_SIZE - MARGIN - GRID_SIZE), 2)
        pygame.draw.line(screen, COLOR_LINE, (MARGIN, MARGIN + i * GRID_SIZE), (WINDOW_SIZE - MARGIN - GRID_SIZE, MARGIN + i * GRID_SIZE), 2)
        
    stars = [3, 7, 11]
    for r in stars:
        for c in stars:
            pygame.draw.circle(screen, COLOR_LINE, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 5)

    # Vẽ quân cờ
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 1:
                pygame.draw.circle(screen, COLOR_BLACK, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), GRID_SIZE // 2 - 2)
            elif board[r, c] == 2:
                pygame.draw.circle(screen, COLOR_WHITE, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), GRID_SIZE // 2 - 2)

    # Đánh dấu nước đi cuối cùng
    if last_action is not None:
        r, c = divmod(last_action, BOARD_SIZE)
        pygame.draw.circle(screen, COLOR_HIGHLIGHT, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 5)

    pygame.display.flip()

def draw_winner(screen, message):
    font = pygame.font.SysFont("Arial", 48, bold=True)
    text = font.render(message, True, (200, 30, 30))
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    overlay.set_alpha(150)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))
    text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()

def load_ai(model_path, device):
    """Hàm tải AI, cấu hình 6 blocks, 64 channels theo đúng model của bạn"""
    network = GomokuNet(board_size=BOARD_SIZE, num_residual_blocks=6, channels=64).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        network.load_state_dict(state_dict)
        network.eval()
        print(f"✅ Tải thành công: {model_path}")
        return network
    except Exception as e:
        print(f"❌ Lỗi tải {model_path}: {e}")
        sys.exit()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE - GRID_SIZE, WINDOW_SIZE - GRID_SIZE))
    pygame.display.set_caption("AI vs AI - Trận Chiến Chế Độ GUI")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    # CÀI ĐẶT 2 ĐẤU THỦ TẠI ĐÂY
    # Bạn có thể trỏ 2 đường dẫn khác nhau nếu muốn test 2 thế hệ
    # ==========================================
    PATH_PLAYER_1 = "model_iter_0013.pt" # Cầm quân Đen
    PATH_PLAYER_2 = "model_iter_0025.pt" # Cầm quân Trắng
    
    net1 = load_ai(PATH_PLAYER_1, device)
    net2 = load_ai(PATH_PLAYER_2, device)

    env = GomokuEnv(board_size=BOARD_SIZE, win_condition=5)
    
    # Num_simulations ở mức 200-400 là đủ để xem nó đánh
    mcts1 = MCTS(env, net1, num_simulations=400)
    mcts2 = MCTS(env, net2, num_simulations=400)
    
    game_over = False
    last_action = None
    draw_board(screen, env.board)
    
    print("\n🔥 TRẬN ĐẤU BẮT ĐẦU! (Xem trên cửa sổ Pygame)")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            current_player = env.current_player
            
            # Chọn MCTS của người đang tới lượt
            active_mcts = mcts1 if current_player == 1 else mcts2
            name = "Quân ĐEN" if current_player == 1 else "Quân TRẮNG"
            pygame.display.set_caption(f"🤖 {name} đang suy nghĩ...") 
            
            with torch.inference_mode():
                action_probs, _ = active_mcts.search(env.board.copy(), current_player, temperature=0.1)
            
            # Chọn nước đi tốt nhất
            action = np.argmax(action_probs)
            
            # Thực thi nước đi
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action
            
            # Cập nhật trí nhớ cho CẢ HAI AI
            mcts1.update_root(action)
            mcts2.update_root(action)
            
            draw_board(screen, env.board, last_action=last_action)
            time.sleep(0.5) # Dừng nửa giây để bạn kịp nhìn nước đi, không bị giật lác
            
            if terminated or truncated:
                winner = info.get("winner", 0)
                if winner == 1:
                    msg = "⬛ QUÂN ĐEN CHIẾN THẮNG!"
                elif winner == 2:
                    msg = "⬜ QUÂN TRẮNG CHIẾN THẮNG!"
                else:
                    msg = "🤝 HÒA NHAU!"
                
                pygame.display.set_caption("Trận đấu kết thúc")
                draw_winner(screen, msg)
                print(msg)
                game_over = True

if __name__ == "__main__":
    main()