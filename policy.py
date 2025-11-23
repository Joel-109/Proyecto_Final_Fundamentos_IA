#policy.py
import numpy as np
import time
from connect4.policy import Policy


# -------------------------------
# Nodo para el árbol de MCTS
# -------------------------------
class Node:
    __slots__ = ("board", "player", "untried", "children",
                 "parent", "W", "N", "Q")

    def __init__(self, board, player, untried, parent=None):
        self.board = board
        self.player = player
        self.untried = list(untried)
        self.children = {}
        self.parent = parent
        self.W = 0.0
        self.N = 0
        self.Q = 0.0


# -------------------------------
# MCTS básico para Connect4
# -------------------------------
class MonteCarloTreeSearchConnectFour:

    def __init__(self, s0: np.ndarray, main_player: int, rng):
        self.s0 = s0
        self.main_player = main_player
        self.rng = rng
        self.c = 1.3

        self.root_node = Node(
            board=s0.copy(),
            player=main_player,
            untried=self.legal_actions(s0),
            parent=None
        )

    def legal_actions(self, s: np.ndarray):
        return [c for c in range(s.shape[1]) if s[0, c] == 0]

    def drop_piece_inplace(self, board, col, player):
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = player
                return r
        return -1

    def check_win_from(self, board, row, col, player):
        if row < 0:
            return False

        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        rows, cols = board.shape

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

    def set_root(self, board, player):
        self.root_node = Node(
            board=board.copy(),
            player=player,
            untried=self.legal_actions(board),
            parent=None
        )

    def run(self, time_limit=0.05):
        start = time.time()
        while time.time() - start < time_limit:
            leaf = self.select()
            _, new = self.expand(leaf)
            final, winner = self.simulate(new)
            self.backpropagate(final, winner)

    def select(self):
        node = self.root_node
        while True:
            if node.untried or not node.children:
                return node

            children = list(node.children.values())
            for c in children:
                if c.N == 0:
                    return c

            parent_log = np.log(max(1, node.N))
            scores = [
                c.Q + self.c * np.sqrt(parent_log / c.N)
                for c in children
            ]
            node = children[np.argmax(scores)]

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

            row = self.drop_piece_inplace(board, chosen, player)

            if self.check_win_from(board, row, chosen, player):
                return node, player

            if np.all(board[0] != 0):
                return node, 0

            player = -player

    def backpropagate(self, leaf, winner):
        node = leaf
        while node:
            node.N += 1

            reward = (
                1.0 if winner == self.main_player else
                -1.0 if winner != 0 else
                0.0
            )

            node.W += reward
            node.Q = node.W / node.N
            node = node.parent


# ===============================
#        POLICY FINAL
# ===============================
class MyPolicy(Policy):

    def __init__(self):
        init_state = np.zeros((6, 7), dtype=int)
        rng = np.random.RandomState(42)

        self.mcts = MonteCarloTreeSearchConnectFour(
            s0=init_state,
            main_player=-1,
            rng=rng
        )

    def mount(self, *args, **kwargs):
        init_state = np.zeros((6, 7), dtype=int)
        rng = np.random.RandomState(42)

        self.mcts = MonteCarloTreeSearchConnectFour(
            s0=init_state,
            main_player=-1,
            rng=rng
        )

    def infer_player(self, s):
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)
        return 1 if ones == negs else -1

    def act(self, s):
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

        best = max(root.children, key=lambda a: root.children[a].N)
        return int(best)

