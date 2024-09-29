import math
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Cell:
    def __init__(self):
        self.parent_i = 0 
        self.parent_j = 0
        self.f = float('inf')  
        self.g = float('inf') 
        self.h = 0

def is_valid(row, col, ROW, COL):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def calculate_h_value(sentence1, sentence2):
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()
    return 1 - cosine_similarity(vectors)[0][1]  

def trace_path(cell_details, dest, ROW, COL):
    path = []
    row = dest[0]
    col = dest[1]

    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    path.append((row, col))
    path.reverse()

    return path

def a_star_search(sentences1, sentences2):
    ROW = len(sentences1)
    COL = len(sentences2)

    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i = 0
    j = 0
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    found_dest = False

    while len(open_list) > 0:
        p = heapq.heappop(open_list)

        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        if i == ROW - 1 and j == COL - 1:
            print("The destination cell is found")
            path = trace_path(cell_details, (ROW - 1, COL - 1), ROW, COL)
            found_dest = True
            break

        directions = [(0, 1), (1, 0), (1, 1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if is_valid(new_i, new_j, ROW, COL) and not closed_list[new_i][new_j]:
                g_new = cell_details[i][j].g + 1.0
                h_new = calculate_h_value(sentences1[new_i], sentences2[new_j])
                f_new = g_new + h_new

                if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                    heapq.heappush(open_list, (f_new, new_i, new_j))
                    cell_details[new_i][new_j].f = f_new
                    cell_details[new_i][new_j].g = g_new
                    cell_details[new_i][new_j].h = h_new
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j

    if found_dest:
        similarity = 0
        path = trace_path(cell_details, (ROW - 1, COL - 1), ROW, COL)
        for i, j in path:
            similarity += 1 - calculate_h_value(sentences1[i], sentences2[j])
        plagiarism_percentage = (similarity / len(path)) * 100
        return plagiarism_percentage
    else:
        print("Failed to find the destination cell")
        return 0

def main():
    para1 = input("Enter the first paragraph: ")
    para2 = input("Enter the second paragraph: ")

    sentences1 = para1.split('. ')
    sentences2 = para2.split('. ')

    plagiarism_percentage = a_star_search(sentences1, sentences2)
    print(f"Plagiarism Percentage: {plagiarism_percentage:.2f}%")

if __name__ == "__main__":
    main()
