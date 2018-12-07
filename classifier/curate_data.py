import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


def main():
	csv_data = open('../1968-01-01-2018-10-15.csv')
	df = pd.read_csv(csv_data)
	csv_data.close()

	id_to_group_data = open('id_to_group.json')
	id_to_group = json.load(id_to_group_data)
	group_to_id_data = open('group_to_id.json')
	group_to_id = json.load(group_to_id_data)
	id_to_group_data.close()
	group_to_id_data.close()

	node_vectors = np.zeros((9599, 128))
	node_vector_data = open('conflict.emd')
	for li in node_vector_data.readlines():
		split_li = li.strip().split(' ')
		split_li = [ float(i) for i in split_li ]
		node_vectors[int(split_li[0]),:] = split_li[1:]
	node_vector_data.close()

	samples = []
	bad_indices = []
	for index, row in df.iterrows():
		if not isinstance(row['actor1'], str) or not isinstance(row['actor2'], str):
			bad_indices.append(index)
			continue
		attacker = row['actor1'].split(';')[0].strip()
		target = row['actor2'].split(';')[0].strip()
		if 'Unidentified' in attacker or attacker == 'Militia (Pro-Government)':
			bad_indices.append(index)
			continue
		if 'Unidentified' in target or target == 'Militia (Pro-Government)':
			bad_indices.append(index)
			continue
		timestamp = [ row['timestamp'] ]
		fatalities = [ row['fatalities'] ]
		latitude = [ row['latitude'] ]
		longitude = [ row['longitude'] ]

		label = [ group_to_id[attacker] ]
		target_vector = node_vectors[group_to_id[target],:].tolist()
		samples.append(label + target_vector + timestamp + fatalities + latitude + longitude)

	samples = np.array(samples)
	df = df.drop(bad_indices)

	inter1 = to_categorical(df['inter1'].values)
	inter2 = to_categorical(df['inter2'].values)
	interaction = to_categorical(df['interaction'].values)

	country_encoder = LabelEncoder()
	integer_encoded = country_encoder.fit_transform(df['country'].values)
	country = to_categorical(integer_encoded)

	samples = np.concatenate((samples, inter1, inter2, interaction, country), axis=1)
	print(samples.shape)
	np.savetxt('samples.csv', samples, delimiter=',')

main()