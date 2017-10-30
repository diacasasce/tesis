# Motor de ajedrez toma de deciciones

import chess
from stockfish import Stockfish
sf = Stockfish()
def mover (mv,ucim,board,sf):
    ucim.append(mv)
    sf.set_position(ucim)
    board.push_uci(mv)
    print(tablero)
def auto(ucim,board,sf):
    mv=sf.get_best_move()
    ucim.append(mv)
    sf.set_position(ucim)
    board.push_uci(mv)
    print(board)
    
tablero = chess.Board()
print(tablero)
