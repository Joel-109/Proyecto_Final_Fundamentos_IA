#policy.py
import numpy as np
from abc import ABC, abstractmethod
from mcts import MonteCarloTreeSearchConnectFour

class Policy(ABC):

    @abstractmethod
    def mount(self) -> None:

        init_state = np.array([[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]])
        
        rng = np.random.RandomState(42)
        mcts = MonteCarloTreeSearchConnectFour(s0=init_state,main_player=-1,rng=rng)
        mcts.run(t=10)
        

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass

