from pathlib import Path
from model_settings import ms
import os
import joblib
ms.find_root(Path())

models_dir = os.path.join(ms.root,ms.trained_models)
models = [f for f in os.listdir(models_dir) if f.find('Legacy')==-1]
model_dir = os.path.join(models_dir,models[0])
pickle = [f for f in os.listdir(model_dir) if f.endswith('.pkl')][0]
model = joblib.load(os.path.join(model_dir,pickle))

from graphviz import Digraph


dot = Digraph(
    graph_attr={
        'dpi': '300',
        'rankdir': 'LR',  # Left-to-right layout
        'nodesep': '0',
        'ranksep': '3'
    }
)

# Add input nodes
for f in model['feature_set']:
    dot.node(f + '_input', f + '_input')

# Add hidden nodes
for f in model['feature_set']:
    dot.node(f + '_hidden', f + '_hidden')

# Add edges between input and hidden nodes
for f in model['feature_set']:
    for g in model['feature_set']:
        if g != f:
            dot.edge(g + '_input', f + '_hidden', arrowsize='0.5')  # Reduce arrow size

# Add edges from hidden nodes to output
for f in model['feature_set']:
    dot.edge(f + '_hidden', 'y', arrowsize='0.5')  # Reduce arrow size

# Add output node
dot.node('y', 'y')

# Render the graph
dot.render(filename='test', format='png')