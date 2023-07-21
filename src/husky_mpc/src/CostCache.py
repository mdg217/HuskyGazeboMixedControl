class CostCache:
    _instance = None  # Attributo privato per memorizzare l'unica istanza

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CostCache, cls).__new__(cls, *args, **kwargs)
            cls._instance.init_cache()
        return cls._instance

    def init_cache(self):
        # Inizializzazione del cache, se necessario
        self.cost = [0, 0, 0]

    def get_cost(self):
        # Ottiene il costo dalla cache
        return self.cost

    def set_cost(self, cost):
        # Imposta il costo nella cache
        self.cost = cost

