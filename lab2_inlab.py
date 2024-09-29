import numpy as np

class Node:
    def __init__(self, parent, state, pcost, hcost):
        self.parent = parent
        self.state = state
        self.pcost = pcost
        self.hcost = hcost
        self.cost = pcost + hcost

    def __hash__(self):
        return hash(''.join(self.state.flatten()))

    def __str__(self):
        return str(self.state)

    def __eq__(self, other):
        return hash(''.join(self.state.flatten())) == hash(''.join(other.state.flatten()))

    def __ne__(self, other):
        return hash(''.join(self.state.flatten())) != hash(''.join(other.state.flatten()))


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, node):
        self.queue.append(node)

    def pop(self):
        state_cost = float('inf')
        index = -1
        for i in range(len(self.queue)):
            if self.queue[i].cost < state_cost:
                state_cost = self.queue[i].cost
                index = i
        return self.queue.pop(index)

    def is_empty(self):
        return len(self.queue) == 0

    def __str__(self):
        l = [i.state for i in self.queue]
        return str(l)

    def __len__(self):
        return len(self.queue)


class Environment:
    def __init__(self, depth=None, goal_state=None):
        self.actions = [1, 2, 3, 4] 
        self.goal_state = goal_state
        self.depth = depth
        self.start_state = self.generate_start_state()

    def generate_start_state(self):
        past_state = self.goal_state
        i = 0
        while i != self.depth:
            new_states = self.get_next_states(past_state)
            choice = np.random.randint(low=0, high=len(new_states))
            if np.array_equal(new_states[choice], past_state):
                continue
            past_state = new_states[choice]
            i += 1
        return past_state

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state

    def get_next_states(self, state):
        space = (0, 0)
        for i in range(3):
            for j in range(3):
                if state[i, j] == '_':
                    space = (i, j)
                    break

        new_states = []
        if space[0] > 0:  
            new_state = np.copy(state)
            new_state[space[0], space[1]] = new_state[space[0] - 1, space[1]]
            new_state[space[0] - 1, space[1]] = '_'
            new_states.append(new_state)

        if space[0] < 2:  
            new_state = np.copy(state)
            new_state[space[0], space[1]] = new_state[space[0] + 1, space[1]]
            new_state[space[0] + 1, space[1]] = '_'
            new_states.append(new_state)

        if space[1] < 2: 
            new_state = np.copy(state)
            new_state[space[0], space[1]] = new_state[space[0], space[1] + 1]
            new_state[space[0], space[1] + 1] = '_'
            new_states.append(new_state)

        if space[1] > 0: 
            new_state = np.copy(state)
            new_state[space[0], space[1]] = new_state[space[0], space[1] - 1]
            new_state[space[0], space[1] - 1] = '_'
            new_states.append(new_state)

        return new_states

    def reached_goal(self, state):
        return np.array_equal(state, self.goal_state)


class Agent:
    def __init__(self, env, heuristic):
        self.frontier = PriorityQueue()
        self.explored = dict()
        self.start_state = env.get_start_state()
        self.goal_state = env.get_goal_state()
        self.env = env
        self.goal_node = None
        self.heuristic = heuristic

    def run(self):
        start_node = Node(None, self.start_state, 0, self.heuristic(self.start_state, self.goal_state))
        self.frontier.push(start_node)

        while not self.frontier.is_empty():
            current_node = self.frontier.pop()
            if self.env.reached_goal(current_node.state):
                self.goal_node = current_node
                break

            self.explored[current_node] = current_node.cost
            for next_state in self.env.get_next_states(current_node.state):
                new_node = Node(current_node, next_state, current_node.pcost + 1,
                                self.heuristic(next_state, self.goal_state))

                if new_node not in self.explored or new_node.cost < self.explored[new_node]:
                    self.frontier.push(new_node)

    def print_nodes(self):
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()

        for state in path:
            print(state)
            print()

def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != '_' and state[i, j] != goal_state[i, j]:
                distance += 1
    return distance

goal = np.array([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '_']])
env = Environment(depth=10, goal_state=goal)

agent = Agent(env, manhattan_distance)
agent.run()
agent.print_nodes()
