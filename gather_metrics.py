import snap
import numpy as np




def load_graph(name):
	if name == 'weighted.txt':
		return snap.LoadEdgeList(snap.PNEANet, 'weighted.txt', 0, 1)
	elif name == 'unweighted.txt':
		return snap.LoadEdgeList(snap.PNGraph, 'unweighted.txt', 0, 1)

def get_highest_deg(G):
	out_deg = [ (node.GetOutDeg(), node.GetId()) for node in G.Nodes() ].sort(reverse=True)
	in_deg = [ (node.GetInDeg(), node.GetId()) for node in G.Nodes() ].sort(reverse=True)


def main():
	G = load_graph('unweighted.txt')
	get_highest_deg(G)


main()