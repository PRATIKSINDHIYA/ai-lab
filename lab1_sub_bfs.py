from collections import deque

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    state = node.state
    empty_index = state.index('_')  # Find the index of the empty stone

    # Possible moves: One step or two steps forward
    moves = [-1, -2, 1, 2]

    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index < len(state):
            new_state = list(state)
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(Node(new_state, node))
    
    return successors

def bfs(start_state, goal_state):
    start_node = Node(start_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0
    while queue:
        node = queue.popleft()
        state_tuple = tuple(node.state)
        
        if state_tuple in visited:
            continue
        
        visited.add(state_tuple)
        nodes_explored +=1
        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]
        return nodes_explored
        
        for successor in get_successors(node):
            queue.append(successor)
    
    return None


start_state = ['E', 'E', 'E', '_', 'W', 'W', 'W']
goal_state = ['W', 'W', 'W', '_', 'E', 'E', 'E']

solution = bfs(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(' '.join(step))
else:
    print("No solution found.")
