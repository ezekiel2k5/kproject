import sys
import os
import time
import random
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx

# --- STEP 1: PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.bmssp_solver import BmsspSolverV2
    from src.graph import Graph
    BMSSP_AVAILABLE = True
    print("BMSSP Library loaded successfully!")
except ImportError as e:
    print(f"Error: {e}")
    BMSSP_AVAILABLE = False

# --- STEP 2: FETCH DATA ---
center_point = (33.578396, 130.42558) 
print("Fetching road network...")
G = ox.graph_from_point(center_point, dist=8000, network_type='drive')

mapping = {node: i for i, node in enumerate(G.nodes())}
source_node = ox.nearest_nodes(G, center_point[1], center_point[0])
source_idx = mapping[source_node]

all_nodes = list(G.nodes)
target_nodes = random.sample(all_nodes, 15)
target_indices = [mapping[t] for t in target_nodes]

# --- STEP 3: INITIALIZE BMSSP GRAPH ---
if BMSSP_AVAILABLE:
    print("Building BMSSP Graph...")
    num_vertices = len(G.nodes)
    bmssp_graph = Graph(num_vertices)
    
    # Auto-detect edge method (robustness for different repo versions)
    add_func = None
    for name in ['add_edge', 'add_weighted_edge', 'add_arc']:
        if hasattr(bmssp_graph, name):
            add_func = getattr(bmssp_graph, name)
            break

    if add_func:
        for u, v, data in G.edges(data=True):
            add_func(mapping[u], mapping[v], data.get('length', 1.0))
    else:
        print("Could not find edge addition method.")
        BMSSP_AVAILABLE = False

# --- STEP 4: BENCHMARK ---
def run_benchmark():
    # 1. Dijkstra
    print("Benchmarking Dijkstra...")
    start_time = time.perf_counter()
    for target in target_nodes:
        try: nx.dijkstra_path(G, source_node, target, weight='length')
        except: pass 
    d_duration = time.perf_counter() - start_time

    # 2. A*
    print("Benchmarking A*...")
    start_time = time.perf_counter()
    for target in target_nodes:
        try: nx.astar_path(G, source_node, target, weight='length')
        except: pass
    a_duration = time.perf_counter() - start_time

    # 3. BMSSP
    b_duration = 0
    if BMSSP_AVAILABLE:
        print("Benchmarking BMSSP...")
        solver = BmsspSolverV2(bmssp_graph)
        start_time = time.perf_counter()
        
        # FIX: Loop through targets because your version requires a 'goal'
        for t_idx in target_indices:
            try:
                # Based on your error, the signature is .solve(start, goal)
                _ = solver.solve(source_idx, t_idx)
            except Exception as e:
                pass
                
        b_duration = time.perf_counter() - start_time
    
    return d_duration, a_duration, b_duration

# --- STEP 5: OUTPUT ---
d_time, a_time, b_time = run_benchmark()

print("\n" + "="*45)
print(f"{'Algorithm':<15} | {'Execution Time':<15}")
print("-" * 45)
print(f"{'Dijkstra':<15} | {d_time:.4f}s")
print(f"{'A*':<15} | {a_time:.4f}s")
if b_time > 0:
    print(f"{'BMSSP':<15} | {b_time:.4f}s")
    # Improvement over the standard Dijkstra
    diff = ((d_time - b_time) / d_time) * 100
    print("-" * 45)
    print(f"BMSSP is {diff:.2f}% faster than Dijkstra")
print("="*45)