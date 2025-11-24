import numpy as np

class DeterministicPolicy:

    def mount(self):
        pass

    def legal_actions(self, s):
        return [c for c in range(7) if s[0, c] == 0]

    def drop_piece(self, board, col, player):
        for r in range(5, -1, -1):
            if board[r, col] == 0:
                board[r, col] = player
                return r
        return None

    def check_win(self, board, row, col, player):
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r,c] == player:
                count += 1
                r += dr; c += dc
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r,c] == player:
                count += 1
                r -= dr; c -= dc
            if count >= 4:
                return True
        return False

    def can_win(self, s, player):
        """devuelve columna que produce victoria inmediata, o None"""
        for col in self.legal_actions(s):
            temp = s.copy()
            row = self.drop_piece(temp, col, player)
            if self.check_win(temp, row, col, player):
                return col
        return None

    def infer_player(self, s):
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)
        if ones == negs:
            return 1
        else:
            return -1

    def act(self, s: np.ndarray) -> int:
        player = self.infer_player(s)
        opp = -player

        legal = self.legal_actions(s)

        # 1. Ganar si es posible
        w = self.can_win(s, player)
        if w is not None:
            return w

        # 2. Bloquear si el rival puede ganar
        b = self.can_win(s, opp)
        if b is not None:
            return b

        # 3. Orden de preferencia determinista
        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in legal:
                return col

        # fallback (no debería pasar)
        return legal[0]
