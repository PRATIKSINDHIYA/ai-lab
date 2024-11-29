import numpy as np
import random
import json

# Function to save data as JSON
def save_data(data):
    with open('data.json', 'w') as outfile:
        json.dump(data.__dict__, outfile)

# MENACE class definition
class Menace:
    def __init__(self):
        self.matchboxes = {}  # Matchboxes to store possible moves
        self.num_win = 0      # Count of wins
        self.num_draw = 0     # Count of draws
        self.num_lose = 0     # Count of losses
        self.moves_played = []  # Store the moves played in the current game

    # Save the MENACE instance
    def save(self):
        save_data(self)

# Validate if the move is legal
def valid_move(board, move):
    return 0 <= move <= 8 and board[move] == " "

# Get available empty spaces on the board
def get_empty_spaces(current_state):
    return np.array([i for i, v in enumerate(current_state) if v == ' '])

# Print the current board state
def print_board(board):
    print("\n    %s | %s | %s\n"
          "  ---+---+---\n"
          "    %s | %s | %s\n"
          "  ---+---+---\n"
          "    %s | %s | %s" % (board[0], board[1], board[2],
                                board[3], board[4], board[5],
                                board[6], board[7], board[8]))

# Check if the game is over and return the game result
def is_game_over(board):
    # Horizontal, vertical, diagonal check
    winning_combos = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical
                      (0, 4, 8), (2, 4, 6)]             # Diagonal

    for combo in winning_combos:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != " ":
            return 10 if board[combo[0]] == "X" else -10

    if len(get_empty_spaces(board)) == 0:
        return 0  # Draw
    return -1  # Game not over

# Get move for MENACE or human
def get_move(board, player):
    if player:  # If it's MENACE's turn
        board_str = ''.join(board)
        if board_str not in player.matchboxes:
            available_moves = [i for i, v in enumerate(board) if v == ' ']
            player.matchboxes[board_str] = available_moves * ((len(available_moves) + 2) // 2)

        beads = player.matchboxes[board_str]
        move = random.choice(beads) if beads else -1
        player.moves_played.append((board_str, move))
        return move
    else:  # Human player
        while True:
            move = int(input("Enter your move (0-8): "))
            if valid_move(board, move):
                return move
            else:
                print("Invalid move. Try again.")

# Update MENACE's learning based on game outcome
def update_menace(player, result):
    if result == "win":
        for (board, move) in player.moves_played:
            player.matchboxes[board].extend([move] * 3)  # Reward for winning
        player.num_win += 1
    elif result == "lose":
        for (board, move) in player.moves_played:
            if move in player.matchboxes[board]:
                player.matchboxes[board].remove(move)  # Penalize for losing
        player.num_lose += 1
    elif result == "draw":
        for (board, move) in player.moves_played:
            player.matchboxes[board].append(move)  # Slight reward for drawing
        player.num_draw += 1
    player.save()

# Train MENACE by playing games between two instances
def train_menace(player1, player2, games=500):
    for _ in range(games):
        player1.moves_played = []
        player2.moves_played = []
        board = [' '] * 9

        while is_game_over(board) == -1:
            move = get_move(board, player1)
            board[move] = 'O'
            if is_game_over(board) != -1:
                break
            move = get_move(board, player2)
            board[move] = 'X'

        result = is_game_over(board)
        if result == 10:
            update_menace(player1, "lose")
        elif result == -10:
            update_menace(player1, "win")
        else:
            update_menace(player1, "draw")

# Initialize MENACE players and load data if available
first_player = Menace()
try:
    with open('data.json', 'r') as f:
        saved_data = json.load(f)
        first_player.matchboxes = saved_data.get("matchboxes", {})
        first_player.num_win = saved_data.get("num_win", 0)
        first_player.num_lose = saved_data.get("num_lose", 0)
        first_player.num_draw = saved_data.get("num_draw", 0)
except FileNotFoundError:
    print("No previous data found. Training MENACE...")

second_player = Menace()
train_menace(first_player, second_player)

# Human vs MENACE game
board = [' '] * 9
print_board(board)

choice = input("Do you want to go first? (y/n): ").lower()
first_player.moves_played = []

if choice == 'y':
    print("You are 'O'.")
    while is_game_over(board) == -1:
        move = get_move(board, None)  # Human move
        board[move] = 'O'
        print_board(board)
        if is_game_over(board) != -1:
            break
        move = get_move(board, first_player)  # MENACE move
        board[move] = 'X'
        print_board(board)
        print(f"\nMENACE moved: {move}")
else:
    print("You are 'X'.")
    while is_game_over(board) == -1:
        move = get_move(board, first_player)  # MENACE move
        board[move] = 'O'
        print_board(board)
        print(f"\nMENACE moved: {move}")
        if is_game_over(board) != -1:
            break
        move = get_move(board, None)  # Human move
        board[move] = 'X'
        print_board(board)

# Determine final result
result = is_game_over(board)
if result == 10:
    update_menace(first_player, "lose")
elif result == -10:
    update_menace(first_player, "win")
else:
    update_menace(first_player, "draw")
