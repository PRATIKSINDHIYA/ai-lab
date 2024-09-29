import numpy as np
import cv2
import random
import math
from scipy.io import loadmat

class JigsawPuzzleSolver:
    def __init__(self, mat_file):
        self.pieces = self.load_images(mat_file)
        self.num_pieces = len(self.pieces)

    def load_images(self, mat_file):
        data = loadmat(mat_file)
        pieces = data['pieces']
        return [pieces[i] for i in range(pieces.shape[0])]  # Dynamic memory allocation via list

    @staticmethod
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    @staticmethod
    def find_edge_matches(piece1, piece2):
        score = cv2.matchTemplate(piece1, piece2, cv2.TM_CCOEFF_NORMED)
        return score[0][0]

    def calculate_cost(self, layout):
        score = 0
        rows = int(self.num_pieces ** 0.5)  # Assuming a square layout
        for r in range(rows):
            for c in range(rows):
                if r > 0:  # Check the piece above
                    score += self.find_edge_matches(
                        self.preprocess_image(layout[r * rows + c]),
                        self.preprocess_image(layout[(r - 1) * rows + c])
                    )
                if c > 0:  # Check the piece to the left
                    score += self.find_edge_matches(
                        self.preprocess_image(layout[r * rows + c]),
                        self.preprocess_image(layout[r * rows + (c - 1)])
                    )
        return -score  # We want to minimize the cost

    def swap_pieces(self, layout):
        new_layout = layout.copy()  # Dynamic memory allocation for new layout
        i, j = random.sample(range(self.num_pieces), 2)  # Randomly select two pieces
        new_layout[i], new_layout[j] = new_layout[j], new_layout[i]  # Swap pieces
        return new_layout

    def simulated_annealing(self, initial_layout, initial_temp=1000, final_temp=1, alpha=0.95):
        current_layout = initial_layout
        current_cost = self.calculate_cost(current_layout)

        temperature = initial_temp

        while temperature > final_temp:
            new_layout = self.swap_pieces(current_layout)
            new_cost = self.calculate_cost(new_layout)

            if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temperature):
                current_layout = new_layout
                current_cost = new_cost

            temperature *= alpha  # Cool down

        return current_layout

    @staticmethod
    def display_puzzle(puzzle):
        rows = int(len(puzzle) ** 0.5)
        cols = rows
        piece_height, piece_width, _ = puzzle[0].shape
        puzzle_image = np.zeros((rows * piece_height, cols * piece_width, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                puzzle_image[r * piece_height:(r + 1) * piece_height,
                             c * piece_width:(c + 1) * piece_width] = puzzle[r * cols + c]

        cv2.imshow("Assembled Puzzle", puzzle_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mat_file = 'scrambled_lena.mat'  # Path to the .mat file containing jigsaw pieces
    puzzle_solver = JigsawPuzzleSolver(mat_file)

    # Start with a random arrangement of pieces
    initial_layout = random.sample(puzzle_solver.pieces, len(puzzle_solver.pieces))

    # Solve the puzzle using simulated annealing
    solved_layout = puzzle_solver.simulated_annealing(initial_layout)
    puzzle_solver.display_puzzle(solved_layout)

