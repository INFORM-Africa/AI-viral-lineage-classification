import abc

class Learner(abc.ABC):
    def __init__(self, X, y, **kwargs):
        pass
    
    @abc.abstractmethod
    def fit(self, X, y):
        pass