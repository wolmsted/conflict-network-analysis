import numpy as np
import os
import snap
import pandas as pd
import copy
import json

def get_spokes(G):
	total_undirected = 0.0
	degree_counts = np.zeros((G.GetNodes() + 2, 3))
	for node in G.Nodes():
		out_spoke = 0
		in_spoke = 0
		undir_spoke = 0
		node_id = node.GetId()
		if node_id >= G.GetNodes():
			continue
		out_deg = node.GetOutDeg()
		for d in range(out_deg):
			nbr = node.GetOutNId(d)
			nbr_node = G.GetNI(nbr)
			if G.IsEdge(nbr, node_id):
				undir_spoke += 1
			else:
				out_spoke += 1
		in_spoke = node.GetDeg() - out_spoke - (2 * undir_spoke)
		degree_counts[node_id,:] = np.array([out_spoke, in_spoke, undir_spoke])
		total_undirected += undir_spoke
	undirected_edges_prop = total_undirected / G.GetEdges()

	print 'Proportion of undirected edges: ' + str(undirected_edges_prop)
	return degree_counts

def partly_undir_rewire(G, spokes):
	spokes_copy = copy.deepcopy(spokes)
	rewired = snap.GenRndGnm(snap.PNGraph, G.GetNodes(), 0)

	# Add undirected edges
	total_undirected = np.sum(spokes_copy[:,2])
	while total_undirected > 1:
		undir_edges = spokes_copy[:,2]
		nonzero_stubs = np.where(undir_edges != 0)[0]
		probs = undir_edges[nonzero_stubs] / total_undirected
		random_stubs = np.random.choice(nonzero_stubs, size=2, p=probs)
		if random_stubs[0] == random_stubs[1]:
			continue
		rewired.AddEdge(random_stubs[0], random_stubs[1])
		rewired.AddEdge(random_stubs[1], random_stubs[0])
		spokes_copy[random_stubs[0],2] -= 1
		spokes_copy[random_stubs[1],2] -= 1
		total_undirected = np.sum(spokes_copy[:,2])

	# Add in/out edges
	total_directed = np.sum(spokes_copy[:,0:2])
	while total_directed > 1:
		out_edges = spokes_copy[:,0]
		in_edges = spokes_copy[:,1]
		nonzero_out_stubs = np.where(out_edges != 0)[0]
		out_probs = out_edges[nonzero_out_stubs] / np.sum(out_edges)
		nonzero_in_stubs = np.where(in_edges != 0)[0]
		in_probs = in_edges[nonzero_in_stubs] / np.sum(in_edges)
		random_out = np.random.choice(nonzero_out_stubs, p=out_probs)
		random_in = np.random.choice(nonzero_in_stubs, p=in_probs)
		if random_out == random_in:
			continue
		rewired.AddEdge(random_out, random_in)
		spokes_copy[random_out,0] -= 1
		spokes_copy[random_in,1] -= 1
		total_directed = np.sum(spokes_copy[:,0:2])
	snap.DelSelfEdges(rewired)
	return rewired

def main():
	# data_list = []
	# for year in range(1997, 2019):
		# print year
	G = snap.LoadEdgeList(snap.PNGraph, 'unweighted.txt', 0, 1)
	os.system('snap/snap/examples/motifs/motifs -i:unweighted.txt -o:orig')
	f = open('orig-counts.tab')
	df = pd.read_csv(f, sep='\t')
	f.close()
	orig_counts = df['Count'].values
	print orig_counts

	spokes = get_spokes(G)
	motifs = np.zeros((10, 13))
	for i in range(10):
		sample_name = 'sample' + str(i)
		print sample_name
		rewired = partly_undir_rewire(G, spokes)
		snap.SaveEdgeList(rewired, sample_name + '.txt')
		os.system('snap/snap/examples/motifs/motifs -i:' + sample_name + '.txt -o:' + sample_name)
		f = open(sample_name + '-counts.tab')
		df = pd.read_csv(f, sep='\t')
		f.close()
		motif_counts = df['Count'].values
		motifs[i,:] = motif_counts
	motifs[0,:] += 1
	os.system('rm -rf sample*')
	mean = np.mean(motifs, axis=0)
	std = np.std(motifs, axis=0)
	z = (orig_counts - mean) / std
	print z
	# z = z / np.linalg.norm(z)
	# 	data_list.append(z.tolist())
	# with open('all_annual_counts.json', 'w') as f:
	# 	json.dump(data_list, f, indent=4)






main()
