import pygame
import sys
import torch
import numpy as np
import time
from GameEnv import GomokuEnv
from network import GomokuNet
from mcts import MCTS

# ==========================================
# CẤU HÌNH GIAO DIỆN VÀ MÀU SẮC
# ==========================================
BOARD_SIZE = 15
GRID_SIZE = 40  
MARGIN = 40     
WINDOW_SIZE = BOARD_SIZE * GRID_SIZE + MARGIN * 2

COLOR_BG = (222, 184, 135)   # Màu nền gỗ
COLOR_LINE = (0, 0, 0)       # Màu đường kẻ caro
COLOR_BLACK = (20, 20, 20)   # Quân đen (Người chơi)
COLOR_WHITE = (235, 235, 235)# Quân trắng (AI)
COLOR_HIGHLIGHT = (255, 0, 0) # Màu đánh dấu nước đi vừa đánh

def draw_board(screen, board, last_action=None):
    """Vẽ bàn cờ, các đường kẻ, hoa thị và các quân cờ"""
    screen.fill(COLOR_BG)
    
    # Vẽ lưới caro
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, COLOR_LINE, 
                         (MARGIN + i * GRID_SIZE, MARGIN), 
                         (MARGIN + i * GRID_SIZE, WINDOW_SIZE - MARGIN - GRID_SIZE), 2)
        pygame.draw.line(screen, COLOR_LINE, 
                         (MARGIN, MARGIN + i * GRID_SIZE), 
                         (WINDOW_SIZE - MARGIN - GRID_SIZE, MARGIN + i * GRID_SIZE), 2)
        
    # Vẽ 5 dấu chấm hoa thị (Thiên nguyên và 4 góc)
    stars = [3, 7, 11]
    for r in stars:
        for c in stars:
            pygame.draw.circle(screen, COLOR_LINE, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 5)

    # Vẽ quân cờ (1 = Đen, 2 = Trắng)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 1:
                pygame.draw.circle(screen, COLOR_BLACK, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), GRID_SIZE // 2 - 2)
            elif board[r, c] == 2:
                pygame.draw.circle(screen, COLOR_WHITE, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), GRID_SIZE // 2 - 2)

    # Đánh dấu nước đi cuối cùng bằng một chấm đỏ nhỏ
    if last_action is not None:
        r, c = divmod(last_action, BOARD_SIZE)
        pygame.draw.circle(screen, COLOR_HIGHLIGHT, (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 5)

    pygame.display.flip()

def draw_winner(screen, message):
    """Vẽ thông báo người chiến thắng nổi bật giữa màn hình"""
    font = pygame.font.SysFont("Arial", 48, bold=True)
    text = font.render(message, True, (200, 30, 30))
    
    # Tạo lớp overlay mờ đi để làm nổi bật dòng chữ
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    overlay.set_alpha(150)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))
    
    text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()

def get_click_pos(pos):
    """Chuyển đổi tọa độ chuột (pixel) sang tọa độ mảng (hàng, cột)"""
    x, y = pos
    col = round((x - MARGIN) / GRID_SIZE)
    row = round((y - MARGIN) / GRID_SIZE)
    return row, col

def load_ai(model_path, device):
    """Hàm tải AI, tương thích với kiến trúc model nhỏ: 6 blocks, 64 channels"""
    network = GomokuNet(board_size=BOARD_SIZE, num_residual_blocks=6, channels=64).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Bóc lớp vỏ DataParallel (trường hợp model được train trên nhiều GPU như Kaggle)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        network.load_state_dict(state_dict)
        network.eval()
        print(f"✅ Tải thành công siêu trí tuệ từ: {model_path}")
        return network
    except Exception as e:
        print(f"❌ Lỗi tải {model_path}: {e}")
        sys.exit()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE - GRID_SIZE, WINDOW_SIZE - GRID_SIZE))
    pygame.display.set_caption("Caro AI: Bạn (Đen) vs AI (Trắng)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    # KHỞI TẠO GAME VÀ AI
    # ==========================================
    MODEL_PATH = "model_iter_0004.pt" 
    network = load_ai(MODEL_PATH, device)
    
    env = GomokuEnv(board_size=BOARD_SIZE, win_condition=5)
    
    # Số lượng mô phỏng của AI. Bạn có thể tăng lên 800 để nó suy nghĩ sâu hơn,
    # nhưng 400 là mức cân bằng tốt giữa tốc độ và chiến thuật.
    mcts = MCTS(env, network, num_simulations=400)
    
    HUMAN_PLAYER = 1 # Người chơi cầm quân Đen (1)
    AI_PLAYER = 2    # AI cầm quân Trắng (2)
    
    game_over = False
    last_action = None
    
    draw_board(screen, env.board)
    print("\nTrận đấu bắt đầu! Bạn đi trước (Quân Đen). Hãy nhấp chuột vào giao diện để đặt quân.")

    # ==========================================
    # VÒNG LẶP SỰ KIỆN CHÍNH
    # ==========================================
    while True:
        # Xử lý các sự kiện chuột/bàn phím
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # 1. LƯỢT CỦA BẠN (Human Player)
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and env.current_player == HUMAN_PLAYER:
                row, col = get_click_pos(pygame.mouse.get_pos())
                
                # Kiểm tra click hợp lệ (không vượt quá bàn cờ và ô chưa có quân nào)
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and env.board[row, col] == 0:
                    action = row * BOARD_SIZE + col
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_action = action
                    
                    # Cập nhật MCTS để AI nhận thức được bạn vừa đánh ở đâu
                    mcts.update_root(action)
                    draw_board(screen, env.board, last_action=last_action)
                    
                    if terminated or truncated:
                        winner = info.get("winner", 0)
                        msg = "🏆 BẠN ĐÃ THẮNG!" if winner == HUMAN_PLAYER else ("🤝 HÒA NHAU!" if winner == 0 else "💀 AI CHIẾN THẮNG!")
                        draw_winner(screen, msg)
                        print(msg)
                        game_over = True

        # 2. LƯỢT CỦA MÁY (AI Player)
        if not game_over and env.current_player == AI_PLAYER:
            pygame.display.set_caption("🤖 AI đang suy nghĩ...") 
            
            with torch.inference_mode():
                # Tham số temperature=0 để AI chọn nước cờ tối ưu tuyệt đối, không đánh bừa
                action_probs, _ = mcts.search(env.board.copy(), env.current_player, temperature=0.0)
            
            action = np.argmax(action_probs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action
            mcts.update_root(action)
            
            pygame.display.set_caption("Caro AI: Bạn (Đen) vs AI (Trắng)")
            draw_board(screen, env.board, last_action=last_action)
            
            if terminated or truncated:
                winner = info.get("winner", 0)
                msg = "💀 AI CHIẾN THẮNG!" if winner != 0 else "🤝 HÒA NHAU!"
                draw_winner(screen, msg)
                print(msg)
                game_over = True

if __name__ == "__main__":
    main()