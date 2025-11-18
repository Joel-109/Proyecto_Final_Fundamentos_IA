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
        self.c = np.sqrt(2)/2
        self.root_node = {
            "main":True,
            "untried": self.legal_actions(s0), 
            "board": s0, 
            "player":main_player, 
            "children":{},
            "Q":0,
            "N":0,
            "W":0,
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

    
    def select(self) -> dict: 
        # Si hay acciones que no se han intentado, no se selecciona nodo.
        if len(self.root_node['untried']) > 0:
            print("Se devolvió raíz")
            return self.root_node
        
        # Aca se debe implementar lo de ucb
        temp_node = self.root_node
        while len(temp_node['untried']) == 0 and len(temp_node['children']) > 0:

            parent_N = max(1,temp_node['N'])
    
            q_values = []
        
            children =list(temp_node['children'].values())

            for node in children:
                if node['N'] == 0:
                    return node
                q_values.append(node['Q']+self.c*np.sqrt(np.log(parent_N)/(node['N'])))

            best_q = np.argmax(q_values)
            temp_node = children[best_q]


        return temp_node

    def expand(self, s: dict) -> tuple:

        rn = self.rng.randint(0,len(s['untried']))
        a = s['untried'][rn]

        new_s = self.step(s['board'].copy(),a,s['player'])

        new_node = {
            "untried": self.legal_actions(new_s),
            "board": new_s,
            "player": s['player']*-1,
            "tie":False,
            "N":0,
            "W":0,
            "Q":0,
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
        is_tie, is_terminal = self.is_terminal_state(s['board'].copy(), last_player)
        if is_terminal:
            if is_tie:
                return s,0
            else:
                return s,last_player
        
        # s representa el jugador enemigo en el tablero 2 (primera iteración del MCTS)
        actual_player = s['player']

        board = s['board'].copy()

        # Comenzamos a hacer la simulación
        while True:
            # seleccionamos accion aleatoria
            actions = self.legal_actions(board)
            rn = self.rng.randint(0,len(actions))
            a = actions[rn]

            
            # cargamos nuevo tablero
            board = self.step(board,a,actual_player)

            last_player = actual_player

            is_tie, is_terminal = self.is_terminal_state(board,last_player)
            if is_terminal:
                if is_tie:
                    return s,0
                else:
                    return s,last_player
                
            actual_player = actual_player*-1

            

    def backpropagate(self, leaf: dict, winner):
        # recordar hacer la excepción de cuando hay empate, reward = 0
        temp_node = leaf

        # El jugador anterior al estado final es el ganador
        reward = 0
        # el estado final se le coloca al jugador que tiene el board final
        while temp_node is not None:
            temp_node['N']+=1

            
            if winner == 0:
                reward = 0
            else:
                if winner == temp_node['player']:
                    reward = 1
                else:
                    reward = -1
                    
            temp_node['W']+=reward
            temp_node['Q']= temp_node['W']/temp_node['N']
                

            temp_node = temp_node['parent'] 




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


