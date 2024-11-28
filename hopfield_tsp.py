import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  

    def train(self, patterns):
        """ Adjust weights based on input patterns. """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  
        np.fill_diagonal(self.weights, 0)  

    def recall(self, pattern, steps=5):
        """ Recall the pattern from a noisy version. """
        for _ in range(steps):
            for i in range(self.size):
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))  
                pattern[i] = 1 if pattern[i] >= 0 else -1  
        return pattern

def setup_tsp_problem(cities):
    """ Prepare binary patterns representing city visits. """
    size = len(cities)
    patterns = []

    # Assign each city a binary pattern
    for i in range(size):
        pattern = np.zeros(size)
        pattern[i] = 1
        patterns.append(pattern)

    return patterns

# example of 10 cities
cities = ['City ' + str(i) for i in range(10)]
hopfield = HopfieldNetwork(10)
patterns = setup_tsp_problem(cities)
hopfield.train(patterns)

# Add noise to the tour and recall the pattern
noisy_tour = np.copy(patterns[0])
noisy_tour[2] = -noisy_tour[2]  
recalled_tour = hopfield.recall(noisy_tour)

# Display noisy and recalled tours
print("Noisy Tour:")
print(noisy_tour)

print("Recalled Tour:")
print(recalled_tour)
