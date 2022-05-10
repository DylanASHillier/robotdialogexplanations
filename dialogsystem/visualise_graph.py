from networkx.readwrite import gpickle
from networkx import draw
from matplotlib import pyplot
from dash import Dash, html
import dash_cytoscape as cyto
## argparse, obtaining name of graph file
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--graph_file", default="extractedgraph0" ,help="name of graph file")
parser.add_argument("--trim_nodes", default=0, type=int, help="number of nodes to trim the graph to, 0 for no trimming")

if __name__ == "__main__":
    args = parser.parse_args()

    cyto.load_extra_layouts()
    graph = gpickle.read_gpickle(f"logs/{args.graph_file}.json")
    if args.trim_nodes > 0:
        graph = graph.subgraph(list(graph.nodes)[:args.trim_nodes])
    node_elements = [
        {'data': {'id': node, 'label': node}} for node in graph.nodes
    ]
    edge_elements = [
        {'data': {'source': u, 'target': v, 'label': label}} for u, v, label in graph.edges.data('label')
    ]
    ### plot in dash
    app = Dash(__name__)
    app.layout = html.Div([
        cyto.Cytoscape(
            id='cytoscape-compound-layout',
            layout={'name': 'spread'},
            style={'width': '100%', 'height': '1000px'},
            elements=node_elements + edge_elements,
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'text-opacity': 0.5,
                        'text-valign': 'center',
                        'text-halign': 'right',
                        'background-color': '#11479e'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 3,
                        'target-arrow-shape': 'triangle',
                        'content': 'data(label)',
                        'line-color': '#9dbaea',
                        'target-arrow-color': '#9dbaea',
                        'curve-style': 'bezier',
                        'target-arrow-fill': 'filled',
                        'label': 'data(label)'
                    }
                }
            ]
        )
    ])

    app.run_server(debug=True)
