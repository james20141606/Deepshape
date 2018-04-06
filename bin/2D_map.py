#! /usr/bin/env python

import numpy as np
import random
length = 40
images = np.zeros((2000,16,length,length))
masks = np.zeros((2000,16,length,length))

for h in range(2000):
    a = np.repeat(np.arange(4),length/4)
    random.shuffle(a)
    for j in range(length):
        for t in range(length):
            for k in range(4):
                for m in range(4):
                    if (a[j] == k)&(a[t] == m):
                        images[h,4*k+m,j,t] = 1

for h in range(2000):
    for i in range(16):
        c= np.concatenate((np.repeat(1,int(0.01*length*length)),np.repeat(0,int(0.99*length*length))))
        mask = c.reshape(length,length)
        random.shuffle(mask)
        masks[h,i] = mask

with h5py.File('output/images') as f:
    f.create_dataset('images',data = images)

with h5py.File('output/masks') as t:
    t.create_dataset('masks',data = masks)
