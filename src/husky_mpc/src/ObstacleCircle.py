import math

class ObstacleCircle:

    def __init__(self, x, y, r):
        self.xc = x
        self.yc = y
        self.r  = r

    def distance(self, x2, y2):
        return math.sqrt((x2 - self.xc)**2 + (y2 - self.yc)**2)

    def intersection(self, center2_x, center2_y, radius2):
        # Calcola la distanza tra i centri delle due circonferenze
        distance_centers = self.distance(center2_x, center2_y)

        # Se la distanza tra i centri delle due circonferenze è maggiore della somma dei loro raggi,
        # allora le due circonferenze sono completamente separate e non si toccano
        if distance_centers > self.r + radius2:
            return False

        # Se la distanza tra i centri delle due circonferenze è minore della differenza dei loro raggi,
        # allora una circonferenza è completamente contenuta nell'altra e si toccano
        if distance_centers < abs(self.r - radius2):
            return True

        # In tutti gli altri casi, le due circonferenze si intersecano
        return True

    def cost(self, x_attuale, y_attuale):
        d = self.distance(x_attuale, y_attuale, self.xc, self.yc)

        if d < self.r:
            # La posizione attuale è all'interno del cerchio, impostiamo un valore di costo positivo
            valore_costo = 1.0  # Puoi personalizzare questo valore come ritieni opportuno
        else:
            # La posizione attuale è all'esterno del cerchio, impostiamo un valore di costo nullo
            valore_costo = 0.0

        return valore_costo
