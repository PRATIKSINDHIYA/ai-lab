import math
import random
import requests
import networkx as nx
import matplotlib.pyplot as plt

# Rajasthan tourist locations with their coordinates (latitude, longitude)
rajasthan_cities = [
    ("Jaipur", 26.9124, 75.7873),
    ("Udaipur", 24.5854, 73.7125),
    ("Jodhpur", 26.2389, 73.0243),
    ("Jaisalmer", 26.9157, 70.9083),
    ("Pushkar", 26.4899, 74.5508),
    ("Ranthambore National Park", 26.0173, 76.5026),
    ("Mount Abu", 24.5926, 72.7156),
    ("Bikaner", 28.0229, 73.3119),
    ("Chittorgarh", 24.8887, 74.6269),
    ("Ajmer", 26.4499, 74.6399),
    ("Bundi", 25.4305, 75.6497),
    ("Kumbhalgarh", 25.1478, 73.5883),
    ("Bharatpur", 27.2152, 77.5030),
    ("Sariska Tiger Reserve", 27.3133, 76.4376),
    ("Ranakpur", 25.1162, 73.4722),
    ("Alwar", 27.5530, 76.6346),
    ("Mandawa", 28.0543, 75.1531),
    ("Shekhawati Region", 27.6094, 75.1376),
    ("Osian", 26.7315, 72.9177),
    ("Barmer", 25.7521, 71.3967)
]

def haversine_distance(city1, city2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = math.radians(city1[1]), math.radians(city1[2])
    lat2, lon2 = math.radians(city2[1]), math.radians(city2[2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(tour, cities, distance_func):
    return sum(distance_func(cities[tour[i]], cities[tour[i-1]]) for i in range(len(tour)))

def simulated_annealing(cities, distance_func, temp, cooling_rate, num_iterations):
    num_cities = len(cities)
    current_tour = list(range(num_cities))
    random.shuffle(current_tour)
    current_distance = total_distance(current_tour, cities, distance_func)
    
    best_tour = current_tour[:]
    best_distance = current_distance
    
    for i in range(num_iterations):
        if temp < 0.1:
            break
        
        new_tour = current_tour[:]
        idx1, idx2 = random.sample(range(num_cities), 2)
        new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
        
        new_distance = total_distance(new_tour, cities, distance_func)
        
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
            current_tour = new_tour
            current_distance = new_distance
            
            if current_distance < best_distance:
                best_tour = current_tour[:]
                best_distance = current_distance
        
        temp *= cooling_rate
    
    return best_tour, best_distance

def read_tsp_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    content = response.text.splitlines()
    
    cities = []
    reading_coords = False
    for line in content:
        if line.strip() == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        if reading_coords:
            if line.strip() == "EOF":
                break
            _, x, y = line.strip().split()
            cities.append((float(x), float(y)))
    return cities

def solve_rajasthan_tour():
    print("Solving Rajasthan Tour Problem:")
    best_tour, best_distance = simulated_annealing(rajasthan_cities, haversine_distance, temp=10000, cooling_rate=0.995, num_iterations=1000000)
    
    print("Best tour of Rajasthan:")
    for i in best_tour:
        print(f"{rajasthan_cities[i][0]}")
    print(f"\nTotal distance: {best_distance:.2f} km")
    print("--------------------")
    
    # Create a graph and visualize the tour
    G = nx.Graph()
    for i in range(len(best_tour)):
        G.add_edge(rajasthan_cities[best_tour[i-1]][0], rajasthan_cities[best_tour[i]][0])
    
    # Visualize the graph
    pos = {city[0]: (city[1], city[2]) for city in rajasthan_cities}
    nx.draw(G, pos, with_labels=True, node_size=100, font_size=8, font_weight='bold')
    plt.show()

def solve_vlsi_tsp(url):
    print(f"Solving VLSI TSP instance: {url}")
    cities = read_tsp_file(url)
    best_tour, best_distance = simulated_annealing(cities, euclidean_distance, temp=10000, cooling_rate=0.995, num_iterations=1000000)
    print(f"Best tour length: {best_distance:.2f}")
    print("--------------------")
    
    # Create a graph and visualize the tour
    G = nx.Graph()
    for i in range(len(best_tour)):
        G.add_edge(i, (i + 1) % len(best_tour))
    
    # Visualize the graph
    pos = {i: (cities[i][0], cities[i][1]) for i in range(len(cities))}
    nx.draw(G, pos, with_labels=False, node_size=100, font_size=8, font_weight='bold')
    plt.show()

# List of VLSI TSP instances
vlsi_instances = [
    "http://www.math.uwaterloo.ca/tsp/vlsi/xqf131.tsp",
    "http://www.math.uwaterloo.ca/tsp/vlsi/xqg237.tsp",
    "http://www.math.uwaterloo.ca/tsp/vlsi/pma343.tsp",
    "http://www.math.uwaterloo.ca/tsp/vlsi/pka379.tsp",
    "http://www.math.uwaterloo.ca/tsp/vlsi/bcl380.tsp"
]

# Solve Rajasthan Tour Problem
solve_rajasthan_tour()

# Solve VLSI TSP instances
for url in vlsi_instances:
    solve_vlsi_tsp(url)