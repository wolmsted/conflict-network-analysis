import snap
import json
import operator

G = snap.LoadEdgeList(snap.PNGraph, 'unweighted.txt')
nodes = snap.TIntFltH()
edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(G, nodes, edges, 1.0)
nodes = [ (n, nodes[n]) for n in nodes ]
sorted_nodes = sorted(nodes, key=operator.itemgetter(1), reverse=True)
f = open('id_to_group.json')
id_to_group = json.load(f)
top_ten = [ (id_to_group[str(n_id)], between) for n_id, between in sorted_nodes if 'Unidentified' not in id_to_group[str(n_id)] ][:10]
print top_ten

