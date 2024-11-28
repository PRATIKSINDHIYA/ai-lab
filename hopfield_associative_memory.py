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
                # Update each neuron by checking its activation against the weighted sum of the others
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))
                # Ensure the state is either 1 or -1 (binary)
                pattern[i] = 1 if pattern[i] >= 0 else -1
        return pattern

    def capacity(self):
        """ Estimate the maximum number of patterns the network can reliably store. """
        return int(self.size / 2)

# Set up a 10x10 Hopfield network (100 neurons)
size = 100
patterns = [np.random.choice([-1, 1], size) for _ in range(3)] 

# Initialize the Hopfield network and train it with the patterns
hopfield = HopfieldNetwork(size)
hopfield.train(patterns)

# Create a noisy version of one of the patterns to test the recall function
noisy_pattern = patterns[0].copy()
noisy_pattern[5] = -noisy_pattern[5]  
recalled_pattern = hopfield.recall(noisy_pattern)

# Function to display patterns as 10x10 images for visualization
def plot_pattern(pattern):
    plt.imshow(pattern.reshape(10, 10), cmap='binary')
    plt.show()

# Display the noisy and the recalled pattern
print("Noisy Pattern:")
plot_pattern(noisy_pattern)

print("Recalled Pattern:")
plot_pattern(recalled_pattern)

# Print out the network's storage capacity
print(f"Network Capacity: {hopfield.capacity()} patterns")
