import numpy as np
from connect4.policy import Policy
from typing import Callable, Dict, Hashable, List, Any, Tuple

State = Hashable
Action = Hashable

class GroupAPolicy(Policy):

    def mount(self) -> None:
        
        init_state = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],]


    def act(self, s: np.ndarray) -> int:
        print("Estado", s)
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))


   
        

        





