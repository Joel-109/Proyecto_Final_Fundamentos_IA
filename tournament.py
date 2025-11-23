import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "extern_policies"))
# ====== IMPORTA TUS POLÍTICAS AQUÍ ======
from Policy_tournament import MyPolicy
from RandomPolicy import RandomPolicy
#from random_policy import RandomPolicy    # ejemplo si tienes otra política
# ========================================


ROWS = 6
COLS = 7


# ---------------------------------
# LÓGICA DEL TABLERO
# ---------------------------------

def legal_actions(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def drop_piece(board, col, player):
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return r
    return -1


def check_win(board, row, col, player):
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in dirs:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r,c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r,c] == player:
            count += 1
            r -= dr
            c -= dc
            
        if count >= 4:
            return True
    return False


# ---------------------------------
# JUGAR UNA PARTIDA ENTRE 2 POLICIES
# ---------------------------------
def play_game(policy_red, policy_yellow):
    """
    Juega una sola partida:
        - policy_red (fichas = 1)
        - policy_yellow (fichas = -1)

    return:
        1  si gana red
        -1 si gana yellow
        0  si empate
    """
    board = np.zeros((ROWS, COLS), dtype=int)

    policy_red.mount()
    policy_yellow.mount()

    player = 1  # rojo comienza

    while True:
        # elegir política según jugador
        policy = policy_red if player == 1 else policy_yellow

        action = policy.act(board.copy())

        # acción ilegal
        if action not in legal_actions(board):
            return -player  # pierde el que movió ilegal

        row = drop_piece(board, action, player)

        # victoria
        if check_win(board, row, action, player):
            return player

        # empate
        if np.all(board[0] != 0):
            return 0

        player = -player


# ---------------------------------
# TORNEO ENTRE DOS POLICIES
# ---------------------------------
def tournament(policyA_class, policyB_class, games=50):

    results = {"A": 0, "B": 0, "draw": 0}

    pA = policyA_class()
    pB = policyB_class()

    for i in range(games):
        if i % 2 == 0:
            # A como red
            res = play_game(pA, pB)
        else:
            # B como red
            res = play_game(pB, pA)

            if res == 1:
                res = -1
            elif res == -1:
                res = 1

        if res == 1:
            results["A"] += 1
        elif res == -1:
            results["B"] += 1
        else:
            results["draw"] += 1


    pA.finalize()

    return results




def run_tournament(policyA_class, policyB_class, games=50):
    """
    Función pensada para ser llamada desde Jupyter Notebook.
    Retorna directamente los resultados sin pedir input ni imprimir.
    """
    return tournament(policyA_class, policyB_class, games)


# ---------------------------------
# EJECUCIÓN DESDE CONSOLA
# ---------------------------------
if __name__ == "__main__":
    print("=== TORNEO ===")

    # Cambia aquí tus políticas si tienes más
    PolicyA = MyPolicy
    # PolicyB = RandomPolicy
    PolicyB = RandomPolicy  # ejemplo: mismo policy contra sí mismo

    N = int(input("Número de juegos a simular: "))

    results = tournament(PolicyA, PolicyB, games=N)

    print("\n=== RESULTADOS ===")
    print(f"Policy A ganó: {results['A']} veces")
    print(f"Policy B ganó: {results['B']} veces")
    print(f"Empates: {results['draw']}")

    saver = MyPolicy()
    saver.finalize()
    print("\n q_values.json guardado exitosamente")