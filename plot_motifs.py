import json
import matplotlib.pyplot as plt

with open('all_annual_counts.json') as f:
	data = json.load(f)
	fig = plt.figure()
	ax = plt.gca()
	plt.xlabel('Year')
	plt.ylabel('Normalized Z-Score')
	plt.title('Z-Score of Each Motif from 1997-2018')
	cm = plt.get_cmap('gist_rainbow')
	years = range(1997, 2019)
	for i in range(13):
		curr_motif = []
		for d in data:
			curr_motif.append(d[i])
		line = ax.plot(years, curr_motif, label='Motif ' + str(i + 1))
		line[0].set_color(cm(i/13.0))
		# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xticks(range(1997, 2019))
	for index, label in enumerate(ax.xaxis.get_ticklabels()):
		if index % 2 != 0:
			label.set_visible(False)
	plt.show()
