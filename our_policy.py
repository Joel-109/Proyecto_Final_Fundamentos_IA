import numpy as np
from policy import Policy
from typing import Callable, Dict, Hashable, List, Any, Tuple

from mcts import MonteCarloTreeSearchConnectFour
State = Hashable
Action = Hashable


class MyPolicy:

    def __init__(self, mcts):
        self.mcts = mcts

    def act(self, s: np.ndarray):

        # 1. Intentar encontrar este tablero en los hijos
        found = False
        for a, child in self.mcts.root_node["children"].items():
            if np.array_equal(child["board"], s):
                print(f">> Root movido al hijo con acción previa {a}")
                self.mcts.root_node = child
                self.mcts.root_node["parent"] = None
                found = True
                break

        # 2. Si no se encuentra, reiniciar el árbol
        if not found:
            print(">> Root NO encontrado. Reset.")
            self.mcts.root_node = {
                "main": True,
                "untried": self.mcts.legal_actions(s),
                "board": s,
                "player": self.mcts.main_player,
                "children": {},
                "Q": 0,
                "N": 0,
                "W": 0,
                "parent": None,
            }

        # 3. Correr MCTS desde este nodo
        self.mcts.run(time_limit=10)

        # 4. Elegir la mejor acción por número de visitas
        best_a = None
        best_N = -1

        for a, child in self.mcts.root_node["children"].items():
            if child["N"] > best_N:
                best_N = child["N"]
                best_a = a

        return best_a
