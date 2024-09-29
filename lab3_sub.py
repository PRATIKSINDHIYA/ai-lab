from string import ascii_lowercase
import random
from itertools import combinations
import numpy as np

print("Enter the number of clauses (m):")
m = int(input())
print("Enter the number of variables in a clause (k):")
k = int(input())
print("Enter the number of variables (n):")
n = int(input())

def create_problem(m, k, n):
    positive_var = list(ascii_lowercase)[:n]
    negative_var = [c.upper() for c in positive_var]
    variables = positive_var + negative_var
    problems = []
    all_combs = list(combinations(variables, k))
    
    for _ in range(m):
        c = random.sample(all_combs, 1)[0]  
        problems.append(c)
        
    return variables, problems

def assignment(variables, n):
    for_positive = list(np.random.choice(2, n))
    for_negative = [abs(1 - i) for i in for_positive]
    assign = for_positive + for_negative
    var_assign = dict(zip(variables, assign))
    return var_assign

def solve(problem, assign):
    count = 0
    for sub in problem:
        l = [assign[val] for val in sub]
        count += any(l)
    return count

def hill_climbing(problem, assign):
    best_assign = assign.copy()
    best_score = solve(problem, assign)
    step = 0
    
    while True:
        improved = False
        
        for key in assign:
            step += 1
            assign[key] = abs(assign[key] - 1)  
            score = solve(problem, assign)
            
            if score > best_score:
                best_score = score
                best_assign = assign.copy()
                improved = True
            
            assign[key] = abs(assign[key] - 1)  
        
        if not improved:
            break

    return best_assign, best_score, step

def beam_search(problem, assign, beam_width):
    current_assigns = [assign]
    step_size = 0
    
    while current_assigns:
        next_assigns = []
        scores = []
        
        for a in current_assigns:
            step_size += 1
            for key in a:
                new_assign = a.copy()
                new_assign[key] = abs(new_assign[key] - 1) 
                score = solve(problem, new_assign)
                next_assigns.append(new_assign)
                scores.append(score)
        
        top_indices = np.argsort(scores)[-beam_width:]
        current_assigns = [next_assigns[i] for i in top_indices]
        
        if any(score == len(problem) for score in scores):
            return current_assigns[0], len(problem), step_size
    
    return current_assigns[0], max(scores), step_size

def variable_neighborhood(problem, assign, max_neighborhood):
    best_assign = assign.copy()
    best_score = solve(problem, assign)
    step = 0
    neighborhood_size = 1
    
    while neighborhood_size <= max_neighborhood:
        current_assign = best_assign.copy()
        improved = False
        
        for key in current_assign:
            step += 1
            current_assign[key] = abs(current_assign[key] - 1)
            score = solve(problem, current_assign)
            
            if score > best_score:
                best_score = score
                best_assign = current_assign.copy()
                improved = True
            
            current_assign[key] = abs(current_assign[key] - 1) 
        
        if not improved:
            neighborhood_size += 1

    return best_assign, best_score, step

variables, problems = create_problem(m, k, n)

for i, problem in enumerate([problems], start=1):
    assign = assignment(variables, n)
    hill_assign, hill_score, hill_steps = hill_climbing(problem, assign)
    beam_assign, beam_score, beam_steps = beam_search(problem, assign, beam_width=3)
    variable_assign, variable_score, variable_steps = variable_neighborhood(problem, assign, max_neighborhood=3)
    
    print(f'Problem {i}: {problem}')
    print(f'Hill Climbing: {hill_assign}, Score: {hill_score}, Steps: {hill_steps}')
    print(f'Beam Search (3): {beam_assign}, Score: {beam_score}, Steps: {beam_steps}')
    print(f'Variable Neighborhood: {variable_assign}, Score: {variable_score}, Steps: {variable_steps}')
    print()
