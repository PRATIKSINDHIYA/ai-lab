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

# Define TSP problem
cities = ['City ' + str(i) for i in range(10)]
size = len(cities)

# Create original tour
original_tour = np.zeros(size)
for i in range(size):
    original_tour[i] = 1  # Example tour visiting all cities

# Initialize and train the network
hopfield = HopfieldNetwork(size)
hopfield.train([original_tour])

# Add noise to the tour
noisy_tour = original_tour.copy()
noise_indices = np.random.choice(size, 3, replace=False)  # Flip 3 bits
noisy_tour[noise_indices] *= -1

# Recall the tour
recalled_tour = hopfield.recall(noisy_tour.copy())

# Display results
print("Original Tour:")
print(original_tour)

print("\nNoisy Tour:")
print(noisy_tour)

print("\nRecalled Tour:")
print(recalled_tour)

# Verify recall correctness
is_recalled_correct = np.array_equal(recalled_tour, original_tour)
print(f"\nRecalled tour matches original: {is_recalled_correct}")
