import os
import glob

files_to_keep = ['data/train.csv', 'data/test.csv']

for dataset in glob.glob('data/*.csv'):
    if dataset not in files_to_keep:
        print('"{}" removed'.format(dataset))
        os.remove(dataset)
