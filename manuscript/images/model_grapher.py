from graphviz import Digraph

dot = Digraph(graph_attr={'dpi': '300'})
dot.node('meta', 'Pricing Function Estimator')
dot.node('Preprocessor', r'<<font face="Cambria Math" point-size="20"><i>X</i></font> (i.e., the feature matrix)>')

dot.node('Scaler', 'Numerical Features')
dot.node('Encoder', 'Categorical Features')
dot.node('estimator', 'Multi-Layer Perceptron')
dot.node('target', r'<<font face="Cambria Math" point-size="20"><i>y</i></font> (i.e.,the target vector)>')

dot.edge('estimator','target',dir='both',label='Training/Estimation')
dot.edge('meta', 'Preprocessor',label='Training/Estimation')
dot.edge('meta', 'target',dir='both')

dot.edge('Preprocessor', 'Scaler', label="Normalization")
dot.edge('Preprocessor', 'Encoder', label="One-Hot-Encoding")
dot.edge('Scaler', 'estimator')
dot.edge('Encoder', 'estimator')

dot.render(filename='MLP', format='png')
