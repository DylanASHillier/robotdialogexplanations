from networkx.readwrite import gpickle
from networkx import draw
from matplotlib import pyplot

graph = gpickle.read_gpickle(f"datasets/ropes/graphtest.json")
print(graph)
print(graph.nodes(data=True))
print([edge[2]["relevance_label"] for edge in graph.edges(data=True)])
# print(list(graph.nodes)[:100])
print([edge for edge in graph.edges(data=True) if edge[2]["relevance_label"]>1])
draw(graph,with_labels=False)
# pyplot.show()
