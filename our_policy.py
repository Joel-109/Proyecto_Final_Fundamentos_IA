#our_policy.py
import numpy as np
from policy import Policy
from typing import Callable, Dict, Hashable, List, Any, Tuple

from mcts import MonteCarloTreeSearchConnectFour
State = Hashable
Action = Hashable


class MyPolicy:
    def __init__(self, mcts: MonteCarloTreeSearchConnectFour, time_limit: float = 1.0):
        """
        mcts: instancia de tu MCTS (se reutiliza entre turnos)
        time_limit: segundos que se deja correr MCTS por jugada
        """
        self.mcts = mcts
        self.time_limit = time_limit

        # pesos ajustados para evaluación final
        self.w_N = 1.0
        self.w_Q = 1.5
        self.w_heur = 2.0

    # --------------------------------------------
    # Utilidades
    # --------------------------------------------
    def _boards_equal(self, b1: np.ndarray, b2: np.ndarray) -> bool:
        return np.array_equal(b1, b2)

    def _center_preference(self, action: int) -> float:
        """Prefiere columnas centrales (3 es la central en 0..6)."""
        return 1.0 / (1 + abs(action - 3))

    def _heuristic_board_score(self, board: np.ndarray, player: int) -> float:
        """
        Evalúa un tablero desde la perspectiva de 'player'.
        Recompensa 2-3-4 en línea y centro. Penaliza amenazas del rival.
        """
        rows, cols = board.shape
        opp = -player
        score = 0.0

        # Center control
        center_col = board[:, cols // 2]
        score += np.sum(center_col == player) * 0.6
        score -= np.sum(center_col == opp) * 0.6

        def eval_window(w):
            s = 0.0
            pc = np.count_nonzero(w == player)
            oc = np.count_nonzero(w == opp)
            ec = np.count_nonzero(w == 0)

            if pc == 4:
                s += 100.0
            elif pc == 3 and ec == 1:
                s += 5.0
            elif pc == 2 and ec == 2:
                s += 2.0

            if oc == 4:
                s -= 100.0
            elif oc == 3 and ec == 1:
                s -= 7.0
            elif oc == 2 and ec == 2:
                s -= 1.0

            return s

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                score += eval_window(board[r, c:c+4])

        # Vertical
        for c in range(cols):
            for r in range(rows - 3):
                score += eval_window(board[r:r+4, c])

        # Diagonal positiva
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r+i, c+i] for i in range(4)])
                score += eval_window(window)

        # Diagonal negativa
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r-i, c+i] for i in range(4)])
                score += eval_window(window)

        return score

    def _evaluate_child(self, action: int, node: dict) -> float:
        """Combina N, Q y heurística en un único score final."""
        N = node.get("N", 0)
        Q = node.get("Q", 0.0)
        board = node.get("board")
        player = self.mcts.main_player

        heur = self._heuristic_board_score(board, player)

        score_N = np.log(1 + N)
        score_Q = Q
        score_heur = heur / 10.0

        center_pref = self._center_preference(action)

        return float(
            self.w_N * score_N +
            self.w_Q * score_Q +
            self.w_heur * score_heur +
            0.1 * center_pref
        )

    # --------------------------------------------
    # Manejo del árbol entre turnos
    # --------------------------------------------
    def _move_root_to_matching_child(self, state: np.ndarray) -> bool:
        """Reubica la raíz al hijo cuyo tablero coincide con el estado actual."""
        root = self.mcts.root_node
        for a, child in list(root.get("children", {}).items()):
            if self._boards_equal(child["board"], state):
                child_copy = child.copy()
                child_copy["parent"] = None
                self.mcts.root_node = child_copy
                return True
        return False

    # --------------------------------------------
    # Política principal
    # --------------------------------------------
    def act(self, s: np.ndarray) -> int:
        """
        Devuelve la mejor acción para el tablero s usando MCTS + heurística.
        """

        # 1. Intentar mover root a un hijo válido
        found = self._move_root_to_matching_child(s)

        # 2. Si no existe, reiniciar con root limpia
        if not found:
            self.mcts.root_node = {
                "main": True,
                "untried": self.mcts.legal_actions(s.copy()),
                "board": s.copy(),
                "player": self.mcts.main_player,
                "children": {},
                "Q": 0,
                "N": 0,
                "W": 0,
                "parent": None,
            }

        # 3. Ejecutar MCTS desde la raíz actual
        self.mcts.run(time_limit=self.time_limit)

        # 4. Evaluar los hijos para elegir la mejor jugada
        best_action = None
        best_score = -1e9
        children = self.mcts.root_node.get("children", {})

        for a, node in children.items():
            score = self._evaluate_child(a, node)
            if score > best_score:
                best_score = score
                best_action = a
            elif score == best_score:
                if node.get("N", 0) > children[best_action].get("N", 0):
                    best_action = a
                elif node.get("Q", 0) > children[best_action].get("Q", 0):
                    best_action = a
                elif self._center_preference(a) > self._center_preference(best_action):
                    best_action = a

        # 5. Si no hay hijos (raro), elegir acción aleatoria válida
        if best_action is None:
            legal = self.mcts.legal_actions(s.copy())
            best_action = int(self.mcts.rng.choice(legal))

        return int(best_action)
