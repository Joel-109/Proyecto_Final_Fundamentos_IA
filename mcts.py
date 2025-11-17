import numpy as np
import time 

class MonteCarloTreeSearchConnectFour:

    def __init__(self, s0 : np.ndarray,main_player : int, rng : np.random.RandomState):
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
        print("Inicializó el árbol")
        self.s0 = s0
        self.Q = {}
        self.N = {}
        self.main_player = main_player
        self.rng = rng
        self.root_node = {
            "untried": self.legal_actions(s0),
            "board": s0,
            "player":main_player,
            "children":{},
            "parent": None
        }

    def eta(self, player):
        return player*-1
    
    def run(self,time_limit=10):
        
        # Primera iteración
        node_s = self.root_node
        start = time.time()
        # Comienza bucle While
        

        while time.time() - start <=10:
            node_s = self.select() 

            node_s, node_s_next = self.expand(node_s)
            leaf_node = self.simulate(node_s_next)
            self.backpropagate(leaf_node)
            print("Atrapado en WHILE en RUN")
        # while tiempo actual < timelimit
        #   select()
        #   expand()
        #   simulate()
        #   backpropagate()

    
    def select(self) -> dict: 
        # Si hay acciones que no se han intentado, no se selecciona nodo.
        if len(self.root_node['untried']) != 0:
            print("Se devolvió raíz")
            return self.root_node
       
        print("Paso por Select P2!!!!")
        # Aca se debe implementar lo de ucb
        temp_node = self.root_node

        while True:
            print("Atrapado en WHILE en SELECT")
            if len(temp_node['children']) ==0:
                return temp_node

            for a,new_node in temp_node['children'].items():
                pass
        actions = self.legal_actions(self.s0)
        # Mientras que el diccionario temporal no esté vacío, la idea es recorrer nodo por nodo
        # Cada nodo será un diccionario
        #while not temp_dict:
        #


    def expand(self, s: dict) -> dict:
                
        rn = self.rng.randint(0,len(s['untried']))
        a = s['untried'][rn]

        new_s = self.step(s['board'],a,s['player'])

        new_node = {
            "untried": self.legal_actions(new_s),
            "board": new_s,
            "player": s['player']*-1,
            "N":0,
            "W":0,
            "children":{},
            "parent": s
        }

        s['untried'].remove(a)

        s['children'][a] = new_node

        return s,new_node


        # Cuando se termine de expandir, se cambia de jugador    

    def simulate(self, s : dict):
        # last_player = -1
        last_player = s['player']*-1
        # Si ya había ganado el jugador anterior
        _, is_terminal = self.is_terminal_state(s['board'], last_player)
        if is_terminal:
            return s
        
        # s representa el jugador enemigo en el tablero 2 (primera iteración del MCTS)
        actual_player = s['player']

        board = s['board']

        temp_node = s
        # Comenzamos a hacer la simulación
        while True:
            # seleccionamos accion aleatoria
            actions = self.legal_actions(board)
            rn = self.rng.randint(0,len(actions))
            a = actions[rn]


            # cargamos nuevo tablero
            board = self.step(board,a,actual_player)

            actual_player = actual_player*-1

            new_node = {
                "untried": self.legal_actions(board),
                "board": board,
                "player": actual_player,
                "N":0,
                "W":0,
                "tie": False,
                "children":{},
                "parent": temp_node
            }

            temp_node['children'][a] = new_node
            temp_node = new_node

            # verificamos que se acabo el juego
            is_tie, is_terminal = self.is_terminal_state(board,actual_player)
            if is_terminal:
                if is_tie:
                    temp_node['tie'] = True
                break
        return temp_node
            

    def backpropagate(self, leaf: dict):
        # recordar hacer la excepción de cuando hay empate, reward = 0
        temp_node = leaf

        # El jugador anterior al estado final es el ganador
        turn = leaf['player']
        winner = leaf['player']*-1

        # el estado final se le coloca al jugador que tiene el board final
        while temp_node['parent'] == None:
            
            if not temp_node['tie']:
                if turn == winner:
                    temp_node['W']+= 1
                else:
                    temp_node['W']-=1          

            temp_node['N']+=1

            parent = temp_node['parent']
            parent['N']+=1      
            




    def legal_actions(self, s : np.ndarray):
        rows, columns = s.shape

        available = []
        for c in range(columns):
            counter = 0
            for r in range(rows):
                if s[r,c] != 0:
                    counter+=1
            
            if counter < 6:
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
            for r in range(3,rows-3):
                if s[r,c] == player and s[r-1,c+1] == player and s[r-2,c+2] == player and s[r-3,c+3] == player:
                    return (False,True)

        if s[0,0] != 0 and s[0,1] != 0 and s[0,2] != 0 and s[0,3] != 0 and s[0,4] != 0 and s[0,5] != 0 and s[0,6] != 0:
            return (True,True)

        return False, False



