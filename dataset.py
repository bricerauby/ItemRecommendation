import urllib.request
import gzip
import shutil
import os


url = 'https://snap.stanford.edu/data/amazon0505.txt.gz'

try : 
    os.mkdir('data')
except FileExistsError:
    print('data folder is already existing')

# download the dataset from the snap url
urllib.request.urlretrieve(url, "data/amazon-meta.txt.gz")

# decompress the file
with open('data/amazon-meta.txt', 'wb') as decompressed_file:
    with gzip.open('data/amazon-meta.txt.gz', 'rb') as compressed_file:
        shutil.copyfileobj(compressed_file, decompressed_file)

