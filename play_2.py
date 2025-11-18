import numpy as np
from mcts import MonteCarloTreeSearchConnectFour
from our_policy import MyPolicy
import numpy as np

def print_board(board):
    """Imprimir el tablero de forma bonita."""
    print("\nTablero:")
    for row in board:
        print(" ".join(["." if x == 0 else ("X" if x == 1 else "O") for x in row]))
    print("0 1 2 3 4 5 6\n")

def play_game(policy, first_player=1):
    """
    policy: objeto MyPolicy
    first_player: quién empieza (1 humano, -1 IA)
    """

    board = np.zeros((6,7), dtype=int)
    current_player = first_player

    print("\n=== INICIA EL JUEGO ===")
    print("Jugador 1 = X")
    print("Jugador -1 = O\n")

    print_board(board)

    while True:

        # ---------------------------
        # TURNO HUMANO
        # ---------------------------
        if current_player == 1:
            while True:
                try:
                    col = int(input("Tu movimiento (0-6): "))
                    if col in policy.mcts.legal_actions(board):
                        break
                    else:
                        print("Columna llena o inválida, usa otra.")
                except:
                    print("Entrada inválida.")
            
            # Aplicamos movimiento humano
            board = policy.mcts.step(board, col, current_player)

            print("\nHiciste movimiento en columna", col)
            print_board(board)

        # ---------------------------
        # TURNO IA (MCTS)
        # ---------------------------
        else:
            print("\nTurno de la IA. Pensando...")

            action = policy.act(board.copy(), current_player)
            board = policy.mcts.step(board, action, current_player)

            print("IA juega en columna", action)
            print_board(board)

        # ---------------------------
        # VERIFICAR VICTORIA
        # ---------------------------
        is_tie, is_terminal = policy.mcts.is_terminal_state(board, current_player)

        if is_terminal:
            if is_tie:
                print("\n¡EMPATE!")
            else:
                if current_player == 1:
                    print("\n¡GANASTE! 🎉")
                else:
                    print("\nPERDISTE, gana la IA 😈")
            break

        # Cambiar turno
        current_player *= -1

    print("\n=== FIN DEL JUEGO ===")


rng = np.random.RandomState(42)
s0 = np.zeros((6,7), dtype=int)

mcts = MonteCarloTreeSearchConnectFour(s0, main_player=-1, rng=rng)
policy = MyPolicy(mcts)

play_game(policy, first_player=1)