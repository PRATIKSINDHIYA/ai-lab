class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    state = node.state
    empty_index = state.index('_')  # Find the index of the empty stone

    # Possible moves: One step or two steps forward or backward
    moves = [-1, -2, 1, 2]

    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index < len(state):
            # Avoid invalid swaps (e.g., jumping over an empty stone)
            if (move == -2 and state[empty_index - 1] == '_') or (move == 2 and state[empty_index + 1] == '_'):
                continue

            new_state = list(state)
            # Swap the empty stone with the rabbit at the new index
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(Node(new_state, node))
    
    return successors

def dfs(start_state, goal_state):
    start_node = Node(start_state)
    stack = [start_node]
    visited = set()
    
    while stack:
        node = stack.pop()
        state_tuple = tuple(node.state)
        
        if state_tuple in visited:
            continue
        
        visited.add(state_tuple)
        
        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]
        
        for successor in get_successors(node):
            stack.append(successor)
    
    return None

# Initial state: ['E', 'E', 'E', '_', 'W', 'W', 'W']
start_state = ['E', 'E', 'E', '_', 'W', 'W', 'W']
goal_state = ['W', 'W', 'W', '_', 'E', 'E', 'E']

solution = dfs(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(' '.join(step))
else:
    print("No solution found.")
