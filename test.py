import pygame
import sys
import torch
import numpy as np
from GameEnv import GomokuEnv
from network import GomokuNet
from mcts import MCTS

# ==========================================
# CẤU HÌNH GIAO DIỆN
# ==========================================
BOARD_SIZE = 15
GRID_SIZE = 40  # Kích thước mỗi ô vuông
MARGIN = 40     # Lề xung quanh bàn cờ
WINDOW_SIZE = BOARD_SIZE * GRID_SIZE + MARGIN * 2

# Màu sắc
COLOR_BG = (222, 184, 135)   # Màu gỗ
COLOR_LINE = (0, 0, 0)       # Màu đường kẻ
COLOR_BLACK = (20, 20, 20)   # Quân đen (X)
COLOR_WHITE = (235, 235, 235)# Quân trắng (O)

def draw_board(screen, board):
    """Vẽ bàn cờ và các quân cờ"""
    screen.fill(COLOR_BG)
    
    # Vẽ lưới
    for i in range(BOARD_SIZE):
        # Đường dọc
        pygame.draw.line(screen, COLOR_LINE, 
                         (MARGIN + i * GRID_SIZE, MARGIN), 
                         (MARGIN + i * GRID_SIZE, WINDOW_SIZE - MARGIN - GRID_SIZE), 2)
        # Đường ngang
        pygame.draw.line(screen, COLOR_LINE, 
                         (MARGIN, MARGIN + i * GRID_SIZE), 
                         (WINDOW_SIZE - MARGIN - GRID_SIZE, MARGIN + i * GRID_SIZE), 2)
        
    # Vẽ các dấu chấm hoa thị (Thiên nguyên, sao)
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

    pygame.display.flip()

def get_click_pos(pos):
    """Chuyển đổi tọa độ chuột (pixel) sang tọa độ mảng (row, col)"""
    x, y = pos
    col = round((x - MARGIN) / GRID_SIZE)
    row = round((y - MARGIN) / GRID_SIZE)
    return row, col

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE - GRID_SIZE, WINDOW_SIZE - GRID_SIZE))
    pygame.display.set_caption("AlphaZero Gomoku")
    
    # ==========================================
    # TẢI MODEL AI
    # ==========================================
    print("Đang tải AI, vui lòng đợi...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "model_iter_0044.pt" # Tên file model của bạn
    
    env = GomokuEnv(board_size=BOARD_SIZE, win_condition=5)
    network = GomokuNet(board_size=BOARD_SIZE, num_residual_blocks=10, channels=128).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        print("Tải AI thành công!")
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return

    mcts = MCTS(env, network, num_simulations=800)
    
    human_player = 1 # Người chơi cầm quân Đen (đi trước)
    game_over = False
    
    draw_board(screen, env.board)

    # ==========================================
    # VÒNG LẶP SỰ KIỆN PYGAME
    # ==========================================
    while True:
        # Xử lý các sự kiện chuột/bàn phím
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Xử lý click chuột của NGƯỜI CHƠI
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and env.current_player == human_player:
                row, col = get_click_pos(pygame.mouse.get_pos())
                
                # Cập nhật nếu click hợp lệ
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and env.board[row, col] == 0:
                    action = row * BOARD_SIZE + col
                    obs, reward, terminated, truncated, info = env.step(action)
                    mcts.update_root(action)
                    draw_board(screen, env.board)
                    
                    if terminated or truncated:
                        print("🎉 TRẬN ĐẤU KẾT THÚC!")
                        game_over = True

        # Lượt của AI
        if not game_over and env.current_player != human_player:
            pygame.display.set_caption("🤖 AlphaZero đang suy nghĩ...") # Đổi tiêu đề cửa sổ
            
            with torch.no_grad():
                action_probs, _ = mcts.search(env.board.copy(), env.current_player)
            action = np.argmax(action_probs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            mcts.update_root(action)
            
            pygame.display.set_caption("AlphaZero Gomoku")
            draw_board(screen, env.board)
            
            if terminated or truncated:
                print("🎉 TRẬN ĐẤU KẾT THÚC!")
                game_over = True

if __name__ == "__main__":
    main()