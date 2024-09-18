import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

### test
def fen_to_vector(fen):
    # piece placement - lowercase for black pieces, uppercase for white pieces. numbers represent consequtive spaces. / represents a new row 
    # active color - whose turn it is, either 'w' or 'b'
    # castling rights - which castling moves are still legal K or k for kingside and Q or q for queenside, '-' if no legal castling moves for either player
    # en passant - if the last move was a pawn moving up two squares, this is the space behind the square for the purposes of en passant
    # halfmove clock - number of moves without a pawn move or piece capture, after 50 of which the game is a draw
    # fullmove number - number of full turns starting at 1, increments after black's move

    # Example FEN of starting position
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    
    parts = re.split(" ", fen)
    piece_placement = re.split("/", parts[0])
    active_color = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_clock = int(parts[5])

    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)
    
    # piece to layer structure taken from reference [1]
    piece_to_layer = {
        'R': 1,
        'N': 2,
        'B': 3,
        'Q': 4,
        'K': 5,
        'P': 6,
        'p': 7,
        'k': 8,
        'q': 9,
        'b': 10,
        'n': 11,
        'r': 12
    }
    
    castling = {
        'K': (7,7),
        'Q': (7,0),
        'k': (0,7),
        'q': (0,0),
    }

    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)
    
    if en_passant != '-':
        bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1
    
    if castling_rights != '-':
        for char in castling_rights:
            bit_vector[0, castling[char][0], castling[char][1]] = 1
    
    if active_color == 'w':
        bit_vector[0, 7, 4] = 1
    else:
        bit_vector[0, 0, 4] = 1

    if halfmove_clock > 0:
        c = 7
        while halfmove_clock > 0:
            bit_vector[0, 3, c] = halfmove_clock%2
            halfmove_clock = halfmove_clock // 2
            c -= 1
            if c < 0:
                break

    if fullmove_clock > 0:
        c = 7
        while fullmove_clock > 0:
            bit_vector[0, 4, c] = fullmove_clock%2
            fullmove_clock = fullmove_clock // 2
            c -= 1
            if c < 0:
                break

    return bit_vector

class ChessDataset(Dataset):
    def __init__(self, data_frame):
        self.fens = torch.from_numpy(np.array([*map(fen_to_vector, data_frame["FEN"])], dtype=np.float32))
        self.evals = torch.Tensor([[x] for x in data_frame["Evaluation"]])
        self._len = len(self.evals)
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        return self.fens[index], self.evals[index]


class TopMovesDataset(Dataset):
    def __init__(self, data_frame, move_directory):
        self.fens = torch.from_numpy(np.array([fen_to_vector(fen) for fen in data_frame["fen"]], dtype=np.float32))
        self.encoded_moves = data_frame["EncodedMoves"]
        self.num_moves = len(move_directory)  # Total number of unique moves
        
    def __len__(self):
        return len(self.encoded_moves)
    
    def __getitem__(self, index):
        fen_vector = self.fens[index]
        encoded_moves = self.encoded_moves.iloc[index]

        # Multi hot vector
        target = torch.zeros(self.num_moves)
        target[encoded_moves] = 1

        return fen_vector, target
def eval_to_int(evaluation):
    try:
        res = int(evaluation)
    except ValueError:
        res = 10000 if evaluation[1] == '+' else -10000
    return res / 100

def encode_moves(top_moves, move_directory):
    return [move_directory[move] for move in top_moves]