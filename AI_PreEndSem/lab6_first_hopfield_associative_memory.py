import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  

    def train(self, patterns):
        """ Train the network with the given set of patterns. """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  
        np.fill_diagonal(self.weights, 0) 
        
    def recall(self, pattern, steps=5):
        """ Recall the closest stored pattern starting from an initial noisy state. """
        for _ in range(steps):
            for i in range(self.size):
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))
                pattern[i] = 1 if pattern[i] >= 0 else -1
        return pattern

# Define size and patterns
size = 100
original_patterns = [np.random.choice([-1, 1], size) for _ in range(3)] 

# Initialize the network and train
hopfield = HopfieldNetwork(size)
hopfield.train(original_patterns)

# Choose an original pattern and create a noisy version
original_pattern = original_patterns[0].copy()
noisy_pattern = original_pattern.copy()
noisy_pattern[5] = -noisy_pattern[5]  # Flip one bit

# Recall the pattern
recalled_pattern = hopfield.recall(noisy_pattern.copy())

# Function to visualize patterns
def plot_pattern(pattern, title):
    plt.imshow(pattern.reshape(10, 10), cmap='binary')
    plt.title(title)
    plt.show()

# Display patterns
print("Original Pattern:")
plot_pattern(original_pattern, "Original Pattern")

print("Noisy Pattern:")
plot_pattern(noisy_pattern, "Noisy Pattern")

print("Recalled Pattern:")
plot_pattern(recalled_pattern, "Recalled Pattern")

# Verify if the recalled pattern matches the original
is_recalled_correct = np.array_equal(recalled_pattern, original_pattern)
print(f"Recalled pattern matches original: {is_recalled_correct}")
