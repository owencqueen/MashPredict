import json
import pandas as pd
import numpy as np
import os

# Specify file name and init boolean
first=True
directory="/mnt/data/mash_sketches/json"
# Iterate over all json
for f in os.listdir(directory):
	fn= os.path.join(directory, f)

	# Open file
	with open(fn, "r") as read_file:

		# Parse JSON
		data=json.load(read_file)

		# Access name,hash attribute
		sample= data['sketches'][0]['name']
		hashes= data['sketches'][0]['hashes']

		# Init df and set column names to hashes
		if first:
		
			onehot= pd.DataFrame(data= np.broadcast_to(1,2000).copy(), columns= [sample], index= hashes)
			first=False
			continue
	
		# Join new hashes to df
		new= pd.DataFrame(data= np.broadcast_to(1,2000).copy(), columns= [sample], index= hashes)
		onehot= pd.concat([onehot, new], axis=1)

# Fill NA and save
onehot= onehot.fillna(0).astype(int)
onehot.to_csv('poplar_onehot.txt', header=True, index= True, sep='\t', mode='a')
