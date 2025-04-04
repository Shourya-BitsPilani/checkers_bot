import random
import copy
import math
def detect_agent_color(self, board):
    PURPLE = (178, 102, 255)
    GREY = (128, 128, 128)
    grid = board.create_board()

    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] in [PURPLE, GREY]:
                return grid[row][col]  
    return PURPLE

def is_agent_starting_at_bottom(board, agent_color):
    grid = board.create_board() 
    if any(cell == agent_color for cell in grid[7]):
        return True 
    else:
        return False 
def evaluate_board(self, board):
    PURPLE = (178, 102, 255)
    GREY = (128, 128, 128)
    agent_color = detect_agent_color(self, board)
    grid = board.create_board()

    if is_agent_starting_at_bottom(board, agent_color):
        positional_weight = [
            [5, 6, 7, 9, 9, 7, 6, 5],  
            [4, 6, 8, 10, 10, 8, 6, 4],
            [4, 6, 9, 11, 11, 9, 6, 4],
            [5, 8, 11, 13, 13, 11, 8, 5],  
            [5, 8, 11, 13, 13, 11, 8, 5],  
            [4, 6, 9, 11, 11, 9, 6, 4],  
            [4, 6, 8, 10, 10, 8, 6, 4],
            [10, 12, 14, 16, 16, 14, 12, 10]  
        ]
    else:
        positional_weight = [
            [10, 12, 14, 16, 16, 14, 12, 10],  
            [4, 6, 8, 10, 10, 8, 6, 4],
            [4, 6, 9, 11, 11, 9, 6, 4],
            [5, 8, 11, 13, 13, 11, 8, 5],  
            [5, 8, 11, 13, 13, 11, 8, 5],  
            [4, 6, 9, 11, 11, 9, 6, 4],  
            [4, 6, 8, 10, 10, 8, 6, 4],
            [5, 6, 7, 9, 9, 7, 6, 5] 
        ]

    agent_pieces, opponent_pieces = 0, 0
    agent_center_control, opponent_center_control = 0, 0
    agent_mobility, opponent_mobility = 0, 0

    opponent_color = GREY if agent_color == PURPLE else PURPLE
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            piece = grid[row][col]
            if piece == agent_color:
                agent_pieces += 1
                agent_center_control += positional_weight[row][col]
                agent_mobility += len(self.getPossibleMoves(board, agent_color))
            elif piece == opponent_color:
                opponent_pieces += 1
                opponent_center_control += positional_weight[row][col]
                opponent_mobility += len(self.getPossibleMoves(board, opponent_color))

    score = (agent_pieces - opponent_pieces) * 10
    score += (agent_center_control - opponent_center_control) * 1.5
    score += (agent_mobility - opponent_mobility) * 0.2
    return score
def apply_move_manually(board, move):
    new_board = copy.deepcopy(board)
    grid = new_board.create_board()

    try:
        start_pos, end_pos = move[:2], move[2:4] if len(move) == 4 else move[0], move[1]
        grid[end_pos[0]][end_pos[1]] = grid[start_pos[0]][start_pos[1]]
        grid[start_pos[0]][start_pos[1]] = None
    except Exception:
        pass
    return new_board

def simulated_annealing(current_score, new_score, temperature):
    if new_score > current_score:
        return True
    else:

        acceptance_prob = math.exp((new_score - current_score) / temperature)
        return random.random() < acceptance_prob
def choose_best_option(move, temperature):
    possible_choices = move[2]
    best_choice = None
    best_value = float('-inf')
    weights = []

    for choice in possible_choices:
        row, col = choice
        value = 5 - abs(3.5 - row) 
        if row == 0 or row == 7:
            value += 10  
        weights.append(value)

        if value > best_value:
            best_value = value
            best_choice = choice
    if random.random() < temperature:
        return random.choices(possible_choices, weights=weights, k=1)[0]
    else:
        return best_choice

def minimax(self, board, depth, maximizing_player, alpha, beta, temperature):
    if depth == 0:
        return evaluate_board(self, board), None

    possible_moves = self.getPossibleMoves(board)
    if not possible_moves:
        return evaluate_board(self, board), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in possible_moves:
            simulated_board = apply_move_manually(board, move)
            eval, _ = minimax(self, simulated_board, depth - 1, False, alpha, beta, temperature)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('-inf')
        best_move = None
        for move in possible_moves:
            simulated_board = apply_move_manually(board, move)
            eval, _ = minimax(self, simulated_board, depth - 1, True, alpha, beta, temperature)
            if eval > min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def group1(self, board):
    base_depth = 3 
    max_depth = 4   
    temperature = 1.0  

    best_move = None
    for current_depth in range(1, max_depth + 1):
        _, best_move = minimax(self, board, current_depth, True, float('-inf'), float('inf'), temperature)
    
    if best_move:
        best_choice = choose_best_option(best_move, temperature)
        temperature *= 0.95  
        return best_move, best_choice
    else:
        self.game.end_turn() 
        return None, None