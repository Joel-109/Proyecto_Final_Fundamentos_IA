#play.py
from mcts import MonteCarloTreeSearchConnectFour
import numpy as np

init_state = np.array([[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]])


rng = np.random.RandomState(42)

mcts = MonteCarloTreeSearchConnectFour(init_state,-1,rng)


mcts.run()

print(mcts.root_node)

# para cada nodo hijo
# identifica el mayor q
# se va por el hijo de mayor q

# root['children'] = {a}
# root = children escogido