#policy.py
import numpy as np
import time
import json
import os
from connect4.policy import Policy


# ---------------------------------
# Nodo ligero para el árbol de MCTS
# ---------------------------------
class Node:
    __slots__ = (
        "board", "player", "untried", "children",
        "parent", "W", "N", "Q"
    )

    def __init__(self, board, player, untried, parent=None):
        self.board = board
        self.player = player
        self.untried = list(untried)
        self.children = {}
        self.parent = parent
        self.W = 0.0
        self.N = 0
        self.Q = 0.0


class MonteCarloTreeSearchConnectFour:

    def __init__(self, s0: np.ndarray, main_player: int, rng: np.random.RandomState,
                 Q_global=None, N_global=None):
        self.s0 = s0
        self.main_player = main_player
        self.rng = rng
        self.c = 1.3

        # tablas globales
        self.Q_global = Q_global if Q_global is not None else {}
        self.N_global = N_global if N_global is not None else {}

        self.root_node = Node(
            board=s0.copy(),
            player=main_player,
            untried=self.legal_actions(s0),
            parent=None
        )

    # codificación única por estado + acción
    def encode(self, board, action):
        return (tuple(board.flatten()), action)

    # ------------- utilidades del tablero -------------
    def legal_actions(self, s: np.ndarray):
        return [c for c in range(s.shape[1]) if s[0, c] == 0]

    def drop_piece_inplace(self, board: np.ndarray, col: int, player: int):
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = player
                return r
        return -1

    def check_win_from(self, board, row, col, player):
        if row < 0:
            return False

        rows, cols = board.shape
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in dirs:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True
        return False

    def is_winning_move(self, board, col, player):
        row = -1
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                row = r
                break
        if row == -1:
            return False

        board[row, col] = player
        win = self.check_win_from(board, row, col, player)
        board[row, col] = 0
        return win

    # ---------------------- ROOT ----------------------
    def set_root(self, board: np.ndarray, player: int):
        self.root_node = Node(
            board=board.copy(),
            player=player,
            untried=self.legal_actions(board),
            parent=None
        )

    # ---------------------- MCTS ----------------------
    def run(self, time_limit=0.05):
        start = time.time()
        while time.time() - start < time_limit:
            node_s = self.select()
            node_s, node_s_next = self.expand(node_s)
            leaf, winner = self.simulate(node_s_next)
            self.backpropagate(leaf, winner)

    def select(self):
        node = self.root_node
        while True:
            if node.untried:
                return node
            if not node.children:
                return node

            children = list(node.children.items())

            # usar Q_global cuando exista
            parent_log = np.log(max(1, node.N))
            best_a, best_child = max(
                children,
                key=lambda kv: (
                    self.Q_global.get(self.encode(node.board, kv[0]), kv[1].Q)
                    + self.c * np.sqrt(
                        parent_log /
                        self.N_global.get(self.encode(node.board, kv[0]), max(1, kv[1].N))
                    )
                )
            )
            return best_child

    def expand(self, node):
        if not node.untried:
            return node, node

        a = int(self.rng.choice(node.untried))
        node.untried.remove(a)

        new_board = node.board.copy()
        self.drop_piece_inplace(new_board, a, node.player)

        new_node = Node(
            board=new_board,
            player=-node.player,
            untried=self.legal_actions(new_board),
            parent=node
        )
        node.children[a] = new_node
        return node, new_node

    def simulate(self, node):
        board = node.board.copy()
        player = node.player

        while True:
            actions = self.legal_actions(board)
            if not actions:
                return node, 0

            chosen = None
            for a in actions:
                if self.is_winning_move(board, a, player):
                    chosen = a
                    break

            if chosen is None:
                opp = -player
                for a in actions:
                    if self.is_winning_move(board, a, opp):
                        chosen = a
                        break

            if chosen is None:
                if 3 in actions:
                    chosen = 3
                else:
                    chosen = int(self.rng.choice(actions))

            col = chosen
            row = self.drop_piece_inplace(board, col, player)

            if self.check_win_from(board, row, col, player):
                return node, player

            if np.all(board[0] != 0):
                return node, 0

            player = -player

    def backpropagate(self, leaf, winner):
        node = leaf
        while node is not None:
            node.N += 1

            reward = (
                1.0 if winner == self.main_player else
                -1.0 if winner != 0 else
                0.0
            )
            node.W += reward
            node.Q = node.W / node.N

            if node.parent is not None:
                for action, child in node.parent.children.items():
                    if child is node:
                        key = self.encode(node.parent.board, action)
                        self.N_global[key] = self.N_global.get(key, 0) + 1
                        self.Q_global[key] = (
                            self.Q_global.get(key, 0.0)
                            + (reward - self.Q_global.get(key, 0.0))
                              / self.N_global[key]
                        )
                        break

            node = node.parent


# ===============================
#        POLICY FINAL
# ===============================
class MyPolicy(Policy):

    def __init__(self):
        self.q_file = "q_values.json"

        # ✅ cargar memoria si existe
        if os.path.exists(self.q_file):
            with open(self.q_file, "r") as f:
                data = json.load(f)
                self.Q_global = {eval(k): v for k, v in data.get("Q", {}).items()}
                self.N_global = {eval(k): v for k, v in data.get("N", {}).items()}
        else:
            self.Q_global = {}
            self.N_global = {}

        init_state = np.zeros((6, 7), dtype=int)
        rng = np.random.RandomState(42)

        self.mcts = MonteCarloTreeSearchConnectFour(
            s0=init_state,
            main_player=-1,
            rng=rng,
            Q_global=self.Q_global,
            N_global=self.N_global
        )

    def finalize(self):
        data = {
            "Q": {str(k): v for k, v in self.Q_global.items()},
            "N": {str(k): v for k, v in self.N_global.items()}
        }
        with open(self.q_file, "w") as f:
            json.dump(data, f, indent=4)

    def infer_player(self, s: np.ndarray) -> int:
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)
        return 1 if ones == negs else -1

    def act(self, s: np.ndarray) -> int:
        player = self.infer_player(s)
        legal = self.mcts.legal_actions(s)

        if not legal:
            return 0

        for a in legal:
            if self.mcts.is_winning_move(s, a, player):
                return a

        opp = -player
        for a in legal:
            if self.mcts.is_winning_move(s, a, opp):
                return a

        if np.all(s == 0) and 3 in legal:
            return 3

        self.mcts.set_root(s, player)
        self.mcts.run(time_limit=0.05)

        root = self.mcts.root_node
        if not root.children:
            return int(self.mcts.rng.choice(legal))

        best_a = max(root.children, key=lambda a: root.children[a].N)
        return int(best_a)

def mount(self, *args, **kwargs):
    init_state = np.zeros((6, 7), dtype=int)
    rng = np.random.RandomState(42)

    self.mcts = MonteCarloTreeSearchConnectFour(
        s0=init_state,
        main_player=-1,
        rng=rng,
        Q_global=self.Q_global,
        N_global=self.N_global
    )
