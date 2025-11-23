#mcts.py
import numpy as np
import time 

class MonteCarloTreeSearchConnectFour:

    def __init__(self, s0: np.ndarray, main_player: int, rng: np.random.RandomState, Q_global=None, N_global=None):
        print("Inicializó el árbol")

        self.s0 = s0
        self.main_player = main_player
        self.rng = rng
        self.c = np.sqrt(2)/2

        self.Q = Q_global if Q_global is not None else {}
        self.N = N_global if N_global is not None else {}

        self.root_node = {
            "main": True,
            "untried": self.legal_actions(s0),
            "board": s0,
            "player": main_player,
            "children": {},
            "Q": 0,
            "N": 0,
            "W": 0,
            "parent": None
        }

    def set_root(self, board, player):
        self.root_node = {
            "main": True,
            "untried": self.legal_actions(board),
            "board": board.copy(),
            "player": player,
            "children": {},
            "Q": 0,
            "N": 0,
            "W": 0,
            "parent": None
        }
    def run(self,time_limit=10):
    
        start = time.time()
        # Comienza bucle While
        

        while time.time() - start <=time_limit:
            node_s = self.select() 

            node_s, node_s_next = self.expand(node_s)
            leaf_node, winner = self.simulate(node_s_next)
            self.backpropagate(leaf_node, winner)

        # while tiempo actual < timelimit
        #   select()
        #   expand()
        #   simulate()
        #   backpropagate()

    
    def select(self):
        node = self.root_node

        while True:

            # 1) Si hay acciones sin intentar → devolver nodo
            if len(node["untried"]) > 0:
                return node

            # 2) Si no tiene hijos → nodo terminal o mal expandido
            if len(node["children"]) == 0:
                return node

            # 3) UCB
            parent_N = max(1, node["N"])
            children = list(node["children"].values())

            # Preferir hijo no visitado
            for c in children:
                if c["N"] == 0:
                    return c

            ucb_scores = [
                c["Q"] + self.c * np.sqrt(np.log(parent_N) / c["N"])
                for c in children
            ]

            node = children[np.argmax(ucb_scores)]



    def expand(self, s: dict) -> tuple:

        # 🎯 1. Si el nodo NO tiene acciones disponibles → NO expandir
        if len(s['untried']) == 0:
            # Esto no debería pasar, pero si pasa devolvemos el nodo tal cual
            return s, s

        # 🎯 2. Acción aleatoria
        rn = self.rng.randint(0,len(s['untried']))
        a = s['untried'][rn]

        new_s = self.step(s['board'].copy(), a, s['player'])

        new_node = {
            "untried": self.legal_actions(new_s),
            "board": new_s,
            "player": s['player'] * -1,
            "tie": False,
            "N": 0,
            "W": 0,
            "Q": 0,
            "children": {},
            "parent": s
        }

        s['untried'].remove(a)
        s['children'][a] = new_node

        return s, new_node

        # Cuando se termine de expandir, se cambia de jugador    

    def simulate(self, node):
        board = node["board"].copy()
        player = node["player"]

        while True:
            actions = self.legal_actions(board)
            a = self.rng.choice(actions)

            board = self.step(board, a, player)

            # Revisar terminalidad DESPUÉS del movimiento
            is_tie, is_terminal = self.is_terminal_state(board, player)

            if is_terminal:
                if is_tie:
                    return node, 0
                else:
                    return node, player

            player *= -1


    def backpropagate(self, leaf, winner):
        node = leaf

        while node is not None:

            node["N"] += 1

            if winner == 0:
                reward = 0
            elif winner == self.main_player:
                reward = 1
            else:
                reward = -1

            node["W"] += reward
            node["Q"] = node["W"] / node["N"]

            node = node["parent"]





    def legal_actions(self, s : np.ndarray):
        rows, columns = s.shape

        available = []
        for c in range(columns):
            counter = 0
            for r in range(rows):
                if s[r,c] != 0:
                    counter+=1
            
            if counter < rows:
                available.append(c)

        return available

    def step(self, s : np.ndarray, a : int,player : int) -> np.ndarray:
        rows, _ = s.shape

        if s[rows-1,a] == 0:
            s[rows-1,a] =  player
            return s

        
        for r in range(rows):
            if s[r,a] != 0:
                s[r-1,a] = player
                break

        return s

    
    def is_terminal_state(self, s : np.ndarray,player) -> tuple:
        
        rows, columns = s.shape
    
        # Verificar Horizontalidad
        for r in range(rows):
            for c in range(columns-3):
                if s[r,c] == player and s[r,c+1] == player and s[r,c+2] == player and s[r,c+3] == player:
                    return False,True
                
        # Verificar Verticalidad
        for c in range(columns):
            for r in range(rows-3):
                if s[r,c] == player and s[r+1,c] == player and s[r+2,c] == player and s[r+3,c] == player:
                    return (False,True)
                
        # Verificar Diagonales (Izquierda)
        for c in range(columns-3):
            for r in range(rows-3):
                if s[r,c] == player and s[r+1,c+1] == player and s[r+2,c+2] == player and s[r+3,c+3] == player:
                    return (False,True)
                
        # Verificar Diagonales (Derecha)
        for c in range(columns-3):
            for r in range(3,rows):
                if s[r,c] == player and s[r-1,c+1] == player and s[r-2,c+2] == player and s[r-3,c+3] == player:
                    return (False,True)

        if s[0,0] != 0 and s[0,1] != 0 and s[0,2] != 0 and s[0,3] != 0 and s[0,4] != 0 and s[0,5] != 0 and s[0,6] != 0:
            return (True,True)

        return False, False


