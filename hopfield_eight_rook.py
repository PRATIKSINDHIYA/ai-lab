import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  

    def train(self, patterns):
        """ Update weights based on the training patterns. """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  
        np.fill_diagonal(self.weights, 0)  

    def recall(self, pattern, steps=5):
        """ Recall a pattern from a noisy input. """
        for _ in range(steps):
            for i in range(self.size):
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))  
                pattern[i] = 1 if pattern[i] >= 0 else -1  
        return pattern

def setup_eight_rook_problem():
    """ Set up the 8-rook problem. """
    size = 64  
    patterns = []

    # Create a valid solution with rooks placed diagonally
    solution = np.zeros(size)
    for i in range(8):
        solution[i*8 + i] = 1  
    patterns.append(solution)

    return patterns

# Initialize and train the Hopfield network
hopfield = HopfieldNetwork(64)
patterns = setup_eight_rook_problem()
hopfield.train(patterns)

# Add noise to the pattern and test recall
noisy_solution = patterns[0].copy()
noisy_solution[3] = -noisy_solution[3] 
recalled_solution = hopfield.recall(noisy_solution)

# Display noisy and recalled solutions
print("Noisy Solution:")
print(noisy_solution.reshape(8, 8))  

print("Recalled Solution:")
print(recalled_solution.reshape(8, 8))  
