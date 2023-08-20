import numpy as np

class uniform_plant():

    def get_plant(self, dim):

        passive_dynamics = np.zeros((dim, dim, dim, dim))

        # Popolamento delle transizioni per gli stati adiacenti
        for row in range(dim):
            for col in range(dim):
                # Stato attuale
                current_state = (row, col)

                # Transizioni possibili: su, giù, sinistra, destra
                possible_transitions = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

                for dr, dc in possible_transitions:
                    next_row = row + dr
                    next_col = col + dc

                    # Verifica se la prossima posizione è all'interno della griglia
                    if 0 <= next_row < dim and 0 <= next_col < dim:
                        # Imposta la probabilità della transizione da current_state a next_state
                        passive_dynamics[row, col, next_row, next_col] = 1.0 / len(possible_transitions)

        # Impostazione delle celle di transizione dallo stato attuale a se stesso a 0
        for row in range(dim):
            for col in range(dim):
                passive_dynamics[row, col, row, col] = 0

        return passive_dynamics