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