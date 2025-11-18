#play_2.py
import numpy as np
import time
from our_policy import MyPolicy
from mcts import MonteCarloTreeSearchConnectFour


# --------------------------------------------------
# Conversión visual: 1 → X, -1 → O, 0 → +
# --------------------------------------------------
def render(board):
    symbols = {1: "X", -1: "O", 0: "+"}

    print("\nTABLERO:")
    for row in board:
        print(" ".join(symbols[val] for val in row))
    print("\n COLUMNAS: 0 1 2 3 4 5 6\n")


def play_game(mcts, policy, human_player=1):
    """
    human_player = 1  → humano juega con X
    human_player = -1 → humano juega con O
    """
    state = mcts.s0.copy()
    current_player = mcts.main_player  # empieza el que definiste en MCTS

    while True:
        render(state)

        # Comprobar terminal
        last_player = -current_player
        tie, terminal = mcts.is_terminal_state(state, last_player)
        if terminal:
            if tie:
                print("¡EMPATE!")
            else:
                winner = "X" if last_player == 1 else "O"
                print(f"¡GANA EL JUGADOR {winner}!")
            break

        # ----------------
        # TURN0 DEL HUMANO
        # ----------------
        if current_player == human_player:
            player_symbol = "X" if human_player == 1 else "O"
            print(f"Turno HUMANO ({player_symbol})")

            actions = mcts.legal_actions(state)
            print("Columnas disponibles:", actions)

            # Entrada segura
            col = input("Seleccione columna: ")

            while not col.isdigit() or int(col) not in actions:
                col = input("Columna inválida, intente de nuevo: ")

            col = int(col)

            # Aplicar acción
            state = mcts.step(state.copy(), col, current_player)

        # ----------------
        # TURNO DEL MCTS
        # ----------------
        else:
            print("\nTurno MCTS... calculando...")

            action = policy.act(state.copy())

            print(f"MCTS juega columna: {action}")

            state = mcts.step(state.copy(), action, current_player)

        # Cambiar jugador
        current_player *= -1


# -------------------------------------
# CONFIGURACIÓN DEL JUEGO
# -------------------------------------
rng = np.random.RandomState(1)

# Estado inicial vacío
s0 = np.zeros((6, 7), dtype=int)

# Crear MCTS
mcts = MonteCarloTreeSearchConnectFour(s0, main_player=1, rng=rng)

# Crear política
policy = MyPolicy(mcts)

# Jugar
play_game(mcts, policy, human_player=1)
