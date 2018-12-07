import snap
import operator
import json

G = snap.LoadEdgeList(snap.PNGraph, 'unweighted.txt')
hubs = snap.TIntFltH()
auth = snap.TIntFltH()
snap.GetHits(G, hubs, auth)
hubs = [ (n, hubs[n]) for n in hubs ]
auth = [ (n, auth[n]) for n in auth ]
sorted_hubs = sorted(hubs, key=operator.itemgetter(1), reverse=True)
f = open('id_to_group.json')
id_to_group = json.load(f)
top_ten = [ (id_to_group[str(n_id)], between) for n_id, between in sorted_hubs if 'Unidentified' not in id_to_group[str(n_id)] ][:10]
print top_ten

sorted_auth = sorted(auth, key=operator.itemgetter(1), reverse=True)
f = open('id_to_group.json')
id_to_group = json.load(f)
top_ten = [ (id_to_group[str(n_id)], between) for n_id, between in sorted_auth[:10] ]
print top_ten