import math

class Expression():
    
    def __init__(self, value):
        self.value = value
    

    def evaluate(self, feed_dict=None):
        raise NotImplementedError
    
    def diff(self, var):
        raise NotImplementedError
    
    def simplify(self):
        return self
    
    def __add__(self, other):
        return Add(self, self._wrap(other))
    


def Add(Expression):

    def __init__(self, left, right):
        super().__init__(None)
        self.left = left
        self.right = right
    
    def evaluate(self, feed_dict=None):
        return self.left.evaluate(feed_dict) + self.right.evaluate(feed_dict)
    
    def diff(self, var):
        return Add(self.left.diff(var), self.right.diff(var))
    
    def __str__(self):
        return f"({self.left}, {self.right})"

