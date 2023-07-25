import numpy as np
from tf import transformations as t
import tf


class ObstacleCircle:

    def __init__(self, obs_name):
        

    def distance(self, x2, y2):
        return np.sqrt((x2 - self.xc)**2 + (y2 - self.yc)**2)

    def are_circles_touching(self, x2, y2, r2):
        # Calcola la distanza tra i centri delle circonferenze
        distance = np.sqrt((x2 - self.xc)**2 + (y2 - self.yc)**2)
        print("La distanza tra le 2 pose è: " + str(distance))

        print(distance <= (self.r + r2))

        # Confronta la distanza con la somma dei raggi delle circonferenze
        return distance <= (self.r + r2)

    
