# Combine two meta files into one
import sys; sys.path.append('../utils')
import pandas as pd

from extract import * 

meta_loc = get_loc_metadata()
meta_climate = get_climate()

meta_loc.index = meta_loc.loc[:,'Geno']
meta_loc = meta_loc.loc[:,(meta_loc.columns != 'Geno')]

print(meta_loc)

meta_climate.index = meta_climate.loc[:,'Geno']
meta_climate = meta_climate.loc[:,(meta_climate.columns != 'Geno')]

print(meta_climate)

left, mega = meta_loc.align(meta_climate, join = 'outer', axis = 0)

print(left)
print(mega)

mega['Binary_class'] = left['Binary_class']
mega['Full_class'] = left['Full_class']

print(mega)

# Filter:
# If full class is empty or all climate variables are empty

col = list(mega.columns)
ind = np.ones(mega.shape[0], dtype = bool)
for c in col:
    if c == 'Binary_class':
        continue
    print(mega.loc[:,c].to_numpy())
    if c == 'Full_class':
        ind &= ~(mega.loc[:,c] == np.array([np.nan]).astype(str)[0])
    else:
        ind &= ~np.isnan(mega.loc[:,c])

mega = mega.loc[ind,(mega.columns != 'Binary_class')]

mega.to_csv('all_meta.csv', sep = '\t')