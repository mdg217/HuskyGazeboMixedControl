import numpy as np
from tf import transformations as t
import tf


class ObstacleCircle:

    def __init__(self, x, y, z, r, T):
        self.r  = r
        self.T  = T

        Tobs = t.concatenate_matrices(t.translation_matrix([x, y, z]),
                               t.quaternion_matrix([0, 0, 0, 0]))
        newTobs = np.dot(t.inverse_matrix(self.T), Tobs)
        self.trans = tf.transformations.translation_from_matrix(newTobs)

        self.xc = self.trans[0]
        self.yc = self.trans[1]


    def distance(self, x2, y2):
        return np.sqrt((x2 - self.xc)**2 + (y2 - self.yc)**2)

    def are_circles_touching(self, x2, y2, r2):
        # Calcola la distanza tra i centri delle circonferenze
        distance = np.sqrt((x2 - self.xc)**2 + (y2 - self.yc)**2)
        print("La distanza tra le 2 pose è: " + str(distance))

        print(distance <= (self.r + r2))

        # Confronta la distanza con la somma dei raggi delle circonferenze
        return distance <= (self.r + r2)

    
