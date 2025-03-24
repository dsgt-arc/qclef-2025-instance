from dwave.cloud import Client
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler
from dwave_networkx.drawing import draw_pegasus

# Connect to the D-Wave cloud
with Client.from_config() as client:
    # Get all solvers
    solvers = client.get_solvers()
    
    print("Available Solvers:")
    for solver in solvers:
        print(f"Solver ID: {solver.id}")
        print(f"Solver Type: {solver.properties.get('category', 'Unknown')}")
        print(f"Qubit Count: {solver.properties.get('num_qubits', 'N/A')}")
        print(f"Topology: {solver.properties.get('topology', 'N/A')}")
        print("-" * 40)

# Select the QPU
sampler = DWaveSampler(solver={'name': 'Advantage_system7.1'})

# Get qubit connectivity
G = sampler.to_networkx_graph()

# Draw the Pegasus topology
plt.figure(figsize=(10, 10))
draw_pegasus(G, with_labels=True, node_size=100)
plt.title("Qubit Connectivity on Advantage_system7.1 (Pegasus)")

# Save the figure to a file
plt.savefig("pegasus_topology.png", dpi=300, bbox_inches='tight')  # Saves as PNG with high resolution

# Close the plot to free memory
plt.close()