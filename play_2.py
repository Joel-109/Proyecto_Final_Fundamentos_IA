import numpy as np
import time
from our_policy import MyPolicy
from mcts import MonteCarloTreeSearchConnectFour

def render(board):
    print("\nTABLERO:")
    for row in board:
        print(row)
    print("COLUMNAS: 0 1 2 3 4 5 6\n")


def play_game(mcts, policy, human_player=1):
    """
    human_player = 1  → humano juega con fichas 1
    human_player = -1 → humano juega con fichas -1
    """
    state = mcts.s0.copy()
    current_player = mcts.main_player  # empieza el main_player

    while True:
        render(state)

        # Comprobar terminal
        last_player = -current_player
        tie, terminal = mcts.is_terminal_state(state, last_player)
        if terminal:
            if tie:
                print("EMPATE!")
            else:
                print(f"Gana el jugador {last_player}")
            break

        # ----------------
        # TURNO DEL HUMANO
        # ----------------
        if current_player == human_player:
            print("Turno HUMANO")
            actions = mcts.legal_actions(state)
            print("Acciones posibles:", actions)

            col = int(input("Seleccione columna: "))

            while col not in actions:
                col = int(input("Columna inválida, intente de nuevo: "))

            # Aplicar acción
            state = mcts.step(state.copy(), col, current_player)

        # ----------------
        # TURNO DEL MCTS
        # ----------------
        else:
            print("\nTurno MCTS... pensando...")

            # Decidir acción con política
            action = policy.act(state.copy())

            print(f"MCTS juega columna: {action}")

            # Aplicar acción
            state = mcts.step(state.copy(), action, current_player)

        # Cambiar jugador
        current_player *= -1

rng = np.random.RandomState(1)

# Estado inicial vacío
s0 = np.zeros((6,7), dtype=int)

# Crear el MCTS
mcts = MonteCarloTreeSearchConnectFour(s0, main_player=1, rng=rng)

# Crear política
policy = MyPolicy(mcts)

# Jugar
play_game(mcts, policy, human_player=1)
