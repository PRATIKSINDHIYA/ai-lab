import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  

    def train(self, patterns):
        """ Train the network on given bipolar patterns (-1 and +1). """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  
        np.fill_diagonal(self.weights, 0)  

    def recall(self, pattern, steps=5):
        """ Recall a pattern with asynchronous updates. """
        for _ in range(steps):
            for i in range(self.size):
                raw_value = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw_value > 0 else -1
        return pattern

# Set up the 8-rook problem
size = 64
original_solution = -np.ones(size)
for i in range(8):
    original_solution[i * 8 + i] = 1  # Place rooks diagonally

# Initialize and train the network
hopfield = HopfieldNetwork(size)
hopfield.train([original_solution])

# Add noise to the original solution
noisy_solution = original_solution.copy()
noise_indices = np.random.choice(size, 5, replace=False)  # Flip 5 random bits
noisy_solution[noise_indices] *= -1

# Recall the solution
recalled_solution = hopfield.recall(noisy_solution.copy())

# Display results
print("Original Solution:")
print(original_solution.reshape(8, 8))

print("\nNoisy Solution:")
print(noisy_solution.reshape(8, 8))

print("\nRecalled Solution:")
print(recalled_solution.reshape(8, 8))

# Verify recall correctness
is_recalled_correct = np.array_equal(recalled_solution, original_solution)
print(f"\nRecalled solution matches original: {is_recalled_correct}")
