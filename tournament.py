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
def tournament_metrics_fast(policyA_class, policyB_class, games=30):

    pA = policyA_class()
    pB = policyB_class()

    wins = 0
    losses = 0
    draws = 0

    lengths = []
    fast_wins = 0
    mid_wins = 0
    late_wins = 0
    sequence = []

    for i in range(games):

        # NO recrear políticas en cada iteración: MUY LENTO
        pA.mount()
        pB.mount()

        if i % 2 == 0:
            res, l = play_game_with_length(pA, pB)
        else:
            res, l = play_game_with_length(pB, pA)
            if res == 1:
                res = -1
            elif res == -1:
                res = 1

        lengths.append(l)

        if l < 10:
            fast_wins += 1
        elif l < 20:
            mid_wins += 1
        else:
            late_wins += 1

        sequence.append(res)

        if res == 1:
            wins += 1
        elif res == -1:
            losses += 1
        else:
            draws += 1

    dominance = (wins - losses) / games
    stability = np.var(sequence)
    momentum = np.mean(sequence[-(games//2):]) - np.mean(sequence[:games//2])

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "lengths": lengths,
        "avg_length": np.mean(lengths),
        "fast_games": fast_wins,
        "mid_games": mid_wins,
        "late_games": late_wins,
        "dominance": dominance,
        "stability": float(stability),
        "momentum": float(momentum),
        "sequence": sequence
    }

def play_game_with_length(policy_red, policy_yellow):
    board = np.zeros((6,7), dtype=int)

    # MONTA UNA SOLA VEZ (como play_game original)
    policy_red.mount()
    policy_yellow.mount()

    player = 1
    turns = 0

    while True:
        policy = policy_red if player == 1 else policy_yellow

        # acción del agente
        action = policy.act(board.copy())
        turns += 1

        if action not in legal_actions(board):
            return -player, turns

        # colocar pieza
        row = drop_piece(board, action, player)

        # victoria
        if check_win(board, row, action, player):
            return player, turns

        # empate
        if np.all(board[0] != 0):
            return 0, turns

        # cambiar jugador
        player = -player


def column_usage(policy_class, games=50):
    """
    Devuelve un array de tamaño 7 indicando
    cuántas veces la política jugó en cada columna.
    Solo mide a policy_class (como jugador RED).
    """
    col_usage = [0] * 7

    for g in range(games):
        board = np.zeros((6,7), dtype=int)
        policy = policy_class()
        policy.mount()
        player = 1  # siempre medimos política como RED

        while True:
            action = policy.act(board.copy())

            # registrar columna usada por el agente
            col_usage[action] += 1

            # ejecutar jugada
            for r in range(5, -1, -1):
                if board[r, action] == 0:
                    board[r, action] = player
                    row = r
                    break

            # victoria / empate
            if check_win(board, row, action, player):
                break
            if np.all(board[0] != 0):
                break

            # el oponente hace jugada random
            legal = [c for c in range(7) if board[0, c] == 0]
            opp_action = np.random.choice(legal)
            for r in range(5, -1, -1):
                if board[r, opp_action] == 0:
                    board[r, opp_action] = -player
                    break

            # victoria o empate del rival
            if check_win(board, r, opp_action, -player):
                break
            if np.all(board[0] != 0):
                break

    return col_usage



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