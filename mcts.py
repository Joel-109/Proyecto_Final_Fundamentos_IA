import numpy as np
import time 

class MonteCarloTreeSearchConnectFour:

    def __init__(self, s0 : np.ndarray,player : int, rng : np.random.RandomState):
        '''
            Estructura de Nodo:
            node = {
                "untried": lista de acciones sin intentar
                "board": es el estado del tablero
                "player": el jugador 
                "W": Cantidad de veces que ha ganado
                "N": Cantidad de visitas a ese nodo
            }
        '''
        self.s0 = s0
        self.Q = {}
        self.N = {}
        self.rng = rng
        self.root_node = {
            "untried": self.legal_actions(s0),
            "board": s0,
            "player":player,
        }

    def eta(self, player):
        return player*-1
    
    def run(self,time_limit=10):
        
        # Primera iteración
        node = self.root_node

        # Comienza bucle While
        node_s = self.select(node) 

        node_s_next = self.expand(node_s)

        self.simulate(node_s_next)
        # while tiempo actual < timelimit
        #   select()
        #   expand()
        #   simulate()
        #   backpropagate()

    
    def select(self,s : dict) -> dict: 

        # Si hay acciones que no se han intentado, no se selecciona nodo.
        if len(s['untried']) != 0:
            return s
       
        # Aca se debe implementar lo de ucb
        
        
        actions = self.legal_actions(self.s0)
        # Mientras que el diccionario temporal no esté vacío, la idea es recorrer nodo por nodo
        # Cada nodo será un diccionario
        #while not temp_dict:
        #


    def select_based_on_ucb(self, s):
        pass

    def expand(self, s: dict) -> dict:
                
        rn = self.rng.randint(0,len(s['untried']))
        a = s['untried'][rn]

        new_s = self.step(s['board'],a,s['player'])

        new_node = {
            "untried": self.legal_actions(new_s),
            "board": new_s,
            "player": s['player']*-1,
            "N":0,
            "W":0
        }

        return new_node


        # Cuando se termine de expandir, se cambia de jugador    

    def simulate(self, s : dict, player : int):
        actual_player = player
        while self.is_terminal_state(s['board'],actual_player):
            rn = self.rng.randint(0,len(s['untried']))
            a = s['untried'][rn]
            actual_player = actual_player

    def backpropagate(self, s:np.ndarray):
        pass


    def legal_actions(self, s : np.ndarray):
        rows, columns = s.shape

        available = []
        for c in range(columns):
            counter = 0
            for r in range(rows):
                if s[r][s] != 0:
                    counter+=1
            
            if counter < 6:
                available.append(c)

        return available

    def step(self, s : np.ndarray, a : int,player : int) -> np.ndarray:
        rows, _ = s.shape

        for r in range(rows):
            if s[r][a] != 0:
                s[r][a] = player
                break

        return s

    def is_terminal_state(self, s : np.ndarray,player):
        
        rows, columns = s.shape
    
        # Verificar Horizontalidad
        for r in range(rows):
            for c in range(columns-3):
                if s[r,c] == player and s[r,c+1] == player and s[r,c+2] == player and s[r,c+3] == player:
                    return player
                
        # Verificar Verticalidad
        for c in range(columns):
            for r in range(rows-3):
                if s[r,c] == player and s[r+1,c] == player and s[r+2,c] == player and s[r+3,c] == player:
                    return player
                
        # Verificar Diagonales (Izquierda)
        for c in range(columns-3):
            for r in range(rows-3):
                if s[r,c] == player and s[r+1,c+1] == player and s[r+2,c+2] == player and s[r+3,c+3] == player:
                    return player
                
        # Verificar Diagonales (Derecha)
        for c in range(columns-3):
            for r in range(3,rows-3):
                if s[r,c] == player and s[r-1,c+1] == player and s[r-2,c+2] == player and s[r-3,c+3] == player:
                    return player

        return 0



