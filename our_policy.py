import numpy as np
from policy import Policy
from typing import Callable, Dict, Hashable, List, Any, Tuple

from mcts import MonteCarloTreeSearchConnectFour
State = Hashable
Action = Hashable

class MyPolicy:

    def __init__(self, mcts: MonteCarloTreeSearchConnectFour):
        self.mcts = mcts

    def act(self, s: np.ndarray, current_player: int) -> int:

        # 1️⃣ Reiniciar el root con el tablero actual
        self.mcts.set_root(s.copy(), current_player)

        # 2️⃣ Ejecutar el MCTS
        self.mcts.run(time_limit=1.0)

        root = self.mcts.root_node

        # 3️⃣ Si NO hay hijos (primer turno del juego) → random válido
        if len(root["children"]) == 0:
            print("⚠️ No había hijos, se juega random")
            return self.mcts.rng.choice(root["untried"])

        # 4️⃣ Elegir por mayor N (robust child)
        best_a = None
        best_N = -1

        for a, child in root["children"].items():
            if child["N"] > best_N:
                best_N = child["N"]
                best_a = a

        print(f"🌟 Acción elegida por mayor N = {best_a}, N = {best_N}")
        return best_a
