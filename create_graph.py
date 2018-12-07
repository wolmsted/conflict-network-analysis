import pandas as pd
import snap
import math
import matplotlib.pyplot as plt
from itertools import permutations
from collections import Counter
import time
import datetime
import numpy as np
import random
import copy
import operator
from scipy.sparse import csc_matrix
import networkx as nx
import os
import json

months = {
    'January' : '1',
    'February' : '2',
    'March' : '3',
    'April' : '4',
    'May' : '5',
    'June' : '6',
    'July' : '7',
    'August' : '8',
    'September' : '9', 
    'October' : '10',
    'November' : '11',
    'December' : '12'
}

def read_file():
	with open('1968-01-01-2018-10-15.csv') as f:
		df = pd.read_csv(f)
		return df

def load_3_subgraphs():
    '''
    Loads a list of all 13 directed 3-subgraphs.
    The list is in the same order as the figure in the HW pdf, but it is
    zero-indexed
    '''
    return [snap.LoadEdgeList(snap.PNGraph, "./subgraphs/{}.txt".format(i), 0, 1) for i in range(13)]

def month_index(date):
	date_broken = date.split(' ')
	month_ind = months[date_broken[1]]
	return date_broken[0] + '/' + month_ind + '/' + date_broken[2]

def add_edges(actor1, actor2, fatalities, civilian_violence, unweighted, weighted, group_to_id, timestamp, weighted_nx):
	if 'Civilians' not in actor1:
		edge_id = weighted.AddEdge(group_to_id[actor1], group_to_id[actor2])
		unweighted.AddEdge(group_to_id[actor1], group_to_id[actor2])
		additional_weight = fatalities
		if additional_weight == 0:
			additional_weight = 0.01
		if weighted_nx.has_edge(group_to_id[actor1], group_to_id[actor2]):
			additional_weight += weighted_nx[group_to_id[actor1]][group_to_id[actor2]]['weight']
		weighted_nx.add_edge(group_to_id[actor1], group_to_id[actor2], weight=additional_weight)
		if fatalities == 0:
			weighted.AddFltAttrDatE(edge_id, 0.1, 'fatalities')	
		else:
			weighted.AddFltAttrDatE(edge_id, float(fatalities), 'fatalities')

	# if not civilian_violence and 'Civilians' not in actor2 :
	# 	edge_id = weighted.AddEdge(group_to_id[actor2], group_to_id[actor1])
	# 	unweighted.AddEdge(group_to_id[actor2], group_to_id[actor1])
	# 	additional_weight = fatalities
	# 	if additional_weight == 0:
	# 		additional_weight = 0.01
	# 	if weighted_nx.has_edge(group_to_id[actor2], group_to_id[actor1]):
	# 		additional_weight += weighted_nx[group_to_id[actor2]][group_to_id[actor1]]['weight']
	# 	weighted_nx.add_edge(group_to_id[actor2], group_to_id[actor1], weight=additional_weight)
	# 	weighted.AddIntAttrDatE(edge_id, timestamp, 'timestamp')

def match(G1, G2):
    '''
    This function compares two graphs of size 3 (number of nodes)
    and checks if they are isomorphic.
    It returns a boolean indicating whether or not they are isomorphic
    You should not need to modify it, but it is also not very elegant...
    '''
    if G1.GetEdges() > G2.GetEdges():
        G = G1
        H = G2
    else:
        G = G2
        H = G1
    # Only checks 6 permutations, since k = 3
    for p in permutations(range(3)):
        edge = G.BegEI()
        matches = True
        while edge < G.EndEI():
            if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
                matches = False
                break
            edge.Next()
        if matches:
            break
    return matches

def count_iso(G, sg, verbose=False):
    '''
    Given a set of 3 node indices in sg, obtains the subgraph from the
    original graph and renumbers the nodes from 0 to 2.
    It then matches this graph with one of the 13 graphs in
    directed_3.
    When it finds a match, it increments the motif_counts by 1 in the relevant
    index

    IMPORTANT: counts are stored in global motif_counts variable.
    It is reset at the beginning of the enumerate_subgraph method.
    '''
    if verbose:
        print(sg)
    nodes = snap.TIntV()
    for NId in sg:
        nodes.Add(NId)
    # This call requires latest version of snap (4.1.0)
    SG = snap.GetSubGraphRenumber(G, nodes)
    for i in range(len(directed_3)):
        if match(directed_3[i], SG):
            motif_counts[i] += 1

def enumerate_subgraph(G, k=3, verbose=False):
    '''
    This is the main function of the ESU algorithm.
    Here, you should iterate over all nodes in the graph,
    find their neighbors with ID greater than the current node
    and issue the recursive call to extend_subgraph in each iteration

    A good idea would be to print a progress report on the cycle over nodes,
    So you get an idea of how long the algorithm needs to run
    '''
    global motif_counts
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)
    for n in G.Nodes():
        n_id = n.GetId()
        degree = n.GetDeg()
        v_ext = set()
        for d in range(degree):
            neighbor = n.GetNbrNId(d)
            if neighbor > n_id:
                v_ext.add(neighbor)
        extend_subgraph(G, k, set([n_id]), v_ext, n_id)


def extend_subgraph(G, k, sg, v_ext, node_id, verbose=False):
    '''
    This is the recursive function in the ESU algorithm
    The base case is already implemented and calls count_iso. You should not
    need to modify this.

    Implement the recursive case.
    '''
    # Base case (you should not need to modify this):
    if len(sg) is k:
        count_iso(G, sg, verbose)
        return
    # Recursive step:
    orig_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        elem = random.sample(v_ext, 1)[0]
        v_ext.remove(elem)

        node = G.GetNI(elem)
        degree = node.GetDeg()
        v_ext_next = set()

        for d in range(degree):
            neighbor = node.GetNbrNId(d)
            if neighbor > node_id and neighbor not in sg and neighbor not in orig_v_ext:
                v_ext_next.add(neighbor)

        extend_subgraph(G, k, sg.union(set([elem])), v_ext_next.union(v_ext), node_id)

def build_graph(df):
	weighted = snap.TNEANet.New()
	unweighted = snap.TNGraph.New()
	weighted_nx = nx.DiGraph()
	group_to_id = {}
	id_to_group = {}
	curr_year = 2018
	for index, row in df.iterrows():
		civilian_violence = (row['event_type'] == 'Violence against civilians')
		actors1 = row['actor1'].split(';') if isinstance(row['actor1'], basestring) else None
		assoc_actors1 = row['assoc_actor_1'].split(';') if isinstance(row['assoc_actor_1'], basestring) else None
		actors2 = row['actor2'].split(';') if isinstance(row['actor2'], basestring) else None
		assoc_actors2 = row['assoc_actor_2'].split(';') if isinstance(row['assoc_actor_2'], basestring) else None
		if actors1 is None or actors2 is None:
			continue
		if assoc_actors1:
			actors1 += assoc_actors1
		if assoc_actors2:
			actors2 += assoc_actors2
		actors1 = [ a.strip() for a in actors1 ]
		actors2 = [ a.strip() for a in actors2 ]
		fatalities = row['fatalities']
		region = row['region']
		timestamp = int(row['timestamp'])
		year = row['year']
		date = month_index(row['event_date'])
		date = int(time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y").timetuple()))

		# if year != curr_year:
		# 	print year, curr_year
		# 	# snap.SaveEdgeListNet(weighted, 'annual-data/weighted' + str(curr_year) + '.txt', 'Weighted edges by fatalities')
		# 	snap.SaveEdgeList(unweighted, 'annual-data/unweighted' + str(curr_year) + '.txt')
		# 	weighted = snap.TNEANet.New()
		# 	unweighted = snap.TNGraph.New()
		# 	group_to_id = {}
		# 	id_to_group = {}
		# 	# for edge in weighted.Edges():
		# 	# 	weighted.DelEdge(edge.GetSrcNId(), edge.GetDstNId())
		# 	# for edge in unweighted.Edges():
		# 	# 	unweighted.DelEdge(edge.GetSrcNId(), edge.GetDstNId())
		# 	curr_year = year

		# if actors[0] == actors[2] or actors[1] == actors[2] or actors[0] == actors[3] or actors[1] == actors[3]:
		# 	continue

		for a in actors1 + actors2:
			if isinstance(a, basestring) and a not in group_to_id:
				group_to_id[a] = unweighted.AddNode()
				weighted.AddNode(group_to_id[a])
				weighted.AddStrAttrDatN(group_to_id[a], a, 'actor')
				weighted.AddStrAttrDatN(group_to_id[a], region, 'region')
				id_to_group[group_to_id[a]] = a
		for a in actors1:
			for b in actors2:
				if isinstance(a, basestring) and isinstance(b, basestring):
					add_edges(a, b, fatalities, civilian_violence, unweighted, weighted, group_to_id, timestamp, weighted_nx)
		# if isinstance(actors[0], basestring):
		# 	if isinstance(actors[2], basestring):
		# 		add_edges(actors[0], actors[2], fatalities, civilian_violence, unweighted, weighted, group_to_id, date, weighted_nx)
		# 	if isinstance(actors[3], basestring):
		# 		add_edges(actors[0], actors[3], fatalities, civilian_violence, unweighted, weighted, group_to_id, date, weighted_nx)
		# if isinstance(actors[1], basestring):
		# 	if isinstance(actors[2], basestring):
		# 		add_edges(actors[1], actors[2], fatalities, civilian_violence, unweighted, weighted, group_to_id, date, weighted_nx)
		# 	if isinstance(actors[3], basestring):
		# 		add_edges(actors[1], actors[3], fatalities, civilian_violence, unweighted, weighted, group_to_id, date, weighted_nx)

	# snap.SaveEdgeListNet(weighted, 'annual-data/weighted' + str(curr_year) + '.txt', 'Weighted edges by fatalities')
	# snap.SaveEdgeList(unweighted, 'annual-data/unweighted' + str(curr_year) + '.txt')
	snap.SaveEdgeListNet(weighted, 'weighted.txt', 'Weighted edges by fatalities')
	snap.SaveEdgeList(unweighted, 'unweighted.txt')
	# with open('group_to_id.json', 'w') as f:
	# 	json.dump(group_to_id, f, indent=4)
	# with open ('id_to_group.json', 'w') as f:
		# json.dump(id_to_group, f, indent=4)
	return unweighted, weighted, weighted_nx, group_to_id, id_to_group

def get_highest_deg(G, group_to_id, id_to_group):
	out_deg = [ (node.GetOutDeg(), node.GetId()) for node in G.Nodes() ]
	in_deg = [ (node.GetInDeg(), node.GetId()) for node in G.Nodes() ]
	out_deg = sorted(out_deg, reverse=True)
	in_deg = sorted(in_deg, reverse=True)
	top_out_deg = [ (id_to_group[node_id], deg) for deg, node_id in out_deg[:10] ]
	print 'Top 10 out unweighted degree actors:'
	print top_out_deg
	top_in_deg = [ (id_to_group[node_id], deg) for deg, node_id in in_deg[:10] ]
	print 'Top 10 in unweighted degree actors:'
	print top_in_deg

	# Plot histogram of in and out degrees
	# deg_out_only, _ = zip(*out_deg)
	# deg_in_only, _ = zip(*in_deg)

	# plt.hist(deg_out_only, bins=20, edgecolor='black')
	# plt.show()

	# plt.hist(deg_in_only, bins=20, edgecolor='black')
	# plt.show()

def get_highest_weighted_deg(G, group_to_id, id_to_group):
	out_deg = np.zeros((G.GetNodes()))
	in_deg = np.zeros(G.GetNodes())

	for n in G.Nodes():
		n_id = n.GetId()
		out_d = n.GetOutDeg()
		in_d = n.GetInDeg()
		for d in range(out_d):
			out_neighbor = n.GetOutNId(d)
			edge = G.GetEI(n_id, out_neighbor)
			out_deg[n_id] += G.GetFltAttrDatE(edge, 'fatalities')
		for d in range(in_d):
			in_neighbor = n.GetInNId(d)
			edge = G.GetEI(in_neighbor, n_id)
			in_deg[n_id] += G.GetFltAttrDatE(edge, 'fatalities')
	in_minus_out = in_deg - out_deg
	in_minus_out = sorted(enumerate(in_minus_out.tolist()), key=operator.itemgetter(1), reverse=True)
	out_minus_in = out_deg - in_deg
	out_minus_in = sorted(enumerate(out_minus_in.tolist()), key=operator.itemgetter(1), reverse=True)
	sorted_out = sorted(enumerate(out_deg.tolist()), key=operator.itemgetter(1), reverse=True)
	sorted_in = sorted(enumerate(in_deg.tolist()), key=operator.itemgetter(1), reverse=True)
	top_out_deg = [ (id_to_group[node_id], deg) for node_id, deg in sorted_out[:10] ]
	print 'Top 10 weighted out degree actors:'
	print top_out_deg
	top_in_deg = [ (id_to_group[node_id], deg) for node_id, deg in sorted_in[:10] ]
	print 'Top 10 weighted in degree actors:'
	print top_in_deg
	print 'Top 10 weighted in - out degreee actors:'
	in_minus_out = [ (id_to_group[node_id], deg) for node_id, deg in in_minus_out[:10] ]
	print in_minus_out
	print 'Top 10 weighted out - in degree actors:'
	out_minus_in = [ (id_to_group[node_id], deg) for node_id, deg in out_minus_in[:10] ]
	print out_minus_in
	# regional_out = {}
	# regional_in = {}

	# for n_id, deg in sorted_out[::-1]:
	# 	region = G.GetStrAttrDatN(n_id, 'region')
	# 	if region in regional_out:
	# 		regional_out[region].append(deg)
	# 	else:
	# 		regional_out[region] = [deg]
	# for n_id, deg in sorted_in[::-1]:
	# 	region = G.GetStrAttrDatN(n_id, 'region')
	# 	if region in regional_in:
	# 		regional_in[region].append(deg)
	# 	else:
	# 		regional_in[region] = [deg]

	# PRankH = snap.TIntFltH()
	# snap.GetPageRank(G, PRankH)
	# sorted_pagerank = sorted([ (id_to_group[item], PRankH[item]) for item in PRankH ], key=operator.itemgetter(1), reverse=True)
	# print 'Top 10 pagerank...'
	# print sorted_pagerank[:10]

	# fig = plt.figure()
	# ax = plt.gca()
	# ax.set_yscale('log')
	# ax.set_xscale('log')

	# for region, val in regional_out.iteritems():
	# 	c = Counter(val)
	# 	counts = [ c[i] for i in range(min(c) , max(c)+1) ]
	# 	ax.scatter(counts, range(len(counts)), edgecolors='none', label=region)
	# ax.legend()
	# plt.show()

	# for region, val in regional_in.iteritems():
	# 	c = Counter(val)
	# 	counts = [ c[i] for i in range(min(c) , max(c)+1) ]
	# 	ax.scatter(counts, range(len(counts)), c='blue', alpha=0.05, edgecolors='none')

	# Plot histogram of in and out degrees
	# _, deg_out_only = zip(*sorted_out)
	# _, deg_in_only = zip(*sorted_in)

	# plt.hist(deg_out_only, bins=20, edgecolor='black')
	# plt.show()

	# plt.hist(deg_in_only, bins=20, edgecolor='black')
	# plt.show()


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

def gather_motifs(G, k=10):
	enumerate_subgraph(G)
	actual_motifs = np.array(motif_counts[:])
	print actual_motifs
	
	spokes = get_spokes(G)
	motifs = np.zeros((10, 13))
	for i in range(k):
		print i
		rewired = partly_undir_rewire(G, spokes)
		enumerate_subgraph(rewired)
		motifs[i,:] = motif_counts[:]
		print motif_counts[:]
	motifs[0,8] += 1

	mean = np.mean(motifs, axis=0)
	std = np.std(motifs, axis=0)
	z = (actual_motifs - mean) / std
	print z
	plt.scatter(range(1,14), z)
	plt.xlabel('Motif Index')
	plt.ylabel('Z-Score')
	plt.title('Z-Score of Each Motif')
	plt.savefig('motifs.png', format='png')

def get_spokes(G):
	total_undirected = 0.0
	degree_counts = np.zeros((G.GetNodes(), 3))
	for node in G.Nodes():
		out_spoke = 0
		in_spoke = 0
		undir_spoke = 0
		node_id = node.GetId()
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

def calculate_pagerank(G, id_to_group, tol=0.0001):
	r = nx.pagerank(G, max_iter=1000, tol=0.00000001)
	sorted_pagerank = sorted([ (id_to_group[key], val) for key, val in r.items() ], key=operator.itemgetter(1), reverse=True)
	print 'Top 10 pagerank...'
	print sorted_pagerank[:10]

if __name__ == "__main__":
	print 'Reading file...'
	df = read_file()
	print 'Building graph...'
	unweighted, weighted, weighted_nx, group_to_id, id_to_group = build_graph(df)
	print 'unweighted nodes'
	print unweighted.GetNodes()
	print 'unweighted edges'
	print unweighted.GetEdges()
	print 'weighted nodes'
	print weighted.GetNodes()
	print 'weighted edges'
	print weighted.GetEdges()
	print 'highest unweighted deg'
	get_highest_deg(unweighted, group_to_id, id_to_group)
	print 'highest weighted deg'
	get_highest_weighted_deg(weighted, group_to_id, id_to_group)
	print ''
	calculate_pagerank(weighted_nx, id_to_group)
	# directed_3 = load_3_subgraphs()
	# motif_counts = [0]*len(directed_3)
	# gather_motifs(unweighted)
