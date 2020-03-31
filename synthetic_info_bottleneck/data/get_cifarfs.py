"""
@author: Arnout Devos
2018/12/06
MIT License

Script for downloading, and reorganizing CIFAR few shot from CIFAR-100 according
to split specifications in Luca et al. '18.
Run this file as follows:
    python get_cifarfs.py

"""

import pickle
import os
import numpy as np
from tqdm import tqdm
import requests
import math
import tarfile,sys
from PIL import Image
import glob
import shutil

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

if not os.path.exists("cifar-100-python.tar.gz"):
    print("Downloading cifar-100-python.tar.gz\n")
    download_file('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz','cifar-100-python.tar.gz')
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

tarname = "cifar-100-python.tar.gz"
print("Untarring: {}".format(tarname))
tar = tarfile.open(tarname)
tar.extractall()
tar.close()

datapath = "cifar-100-python"

print("Extracting jpg images and classes from pickle files")

# in CIFAR 100, the files are given in a train and test format
for batch in ['test','train']:

    print("Handling pickle file: {}".format(batch))

    # Create variable which is the exact path to the file
    fpath = os.path.join(datapath, batch)

    # Unpickle the file, and its metadata (classnames)
    f = open(fpath, 'rb')
    labels = pickle.load(open(os.path.join(datapath, 'meta'), 'rb'), encoding="ASCII")
    d = pickle.load(f, encoding='bytes')

    # decode utf8 encoded keys, and copy files into new dictionary d_decoded
    d_decoded = {}
    for k, v in d.items():
          d_decoded[k.decode('utf8')] = v

    d = d_decoded
    f.close()

    #for i, filename in enumerate(d['filenames']):
    i=0
    for filename in tqdm(d['filenames']):
        folder = os.path.join('images',
                              labels['fine_label_names'][d['fine_labels'][i]]
        )

        #batch,
        #labels['coarse_label_names'][d['coarse_labels'][i]],
        #labels['fine_label_names'][d['fine_labels'][i]]

        png_path = os.path.join(folder, filename.decode())
        jpg_path = os.path.splitext(png_path)[0]+".jpg"

        if os.path.exists(jpg_path):
            continue
        else:
            os.makedirs(folder, exist_ok=True)
            q = d['data'][i]
            with open(jpg_path, 'wb') as outfile:
                #png.from_array(q.reshape((32, 32, 3), order='F').swapaxes(0,1), mode='RGB').save(outfile)
                img = Image.fromarray(q.reshape((32, 32, 3), order='F').swapaxes(0,1), 'RGB')
                img.save(outfile)

        i+=1

print("Removing pickle files")
shutil.rmtree('cifar-100-python', ignore_errors=True)

print("Depending on the split files, organize train, val and test sets")
for datatype in ['train', 'val', 'test']:
    os.makedirs(os.path.join('cifar-fs', datatype), exist_ok=True)
    with open(os.path.join('cifar-fs-splits', datatype + '.txt'), 'r') as f:
        content = f.readlines()
    # Remove whitespace characters like `\n` at the end of each line
    classes = [x.strip() for x in content]

    for img_class in classes:
        if os.path.exists(os.path.join('cifar-fs', datatype, img_class)):
            continue
        else:
            cur_dir = os.path.join('cifar-fs', datatype)
            os.makedirs(cur_dir, exist_ok=True)
            os.system('mv images/' + img_class + ' ' + cur_dir)

print("Removing original CIFAR 100 images")
shutil.rmtree('images', ignore_errors=True)

print("Removing tar file")
os.remove('cifar-100-python.tar.gz')
