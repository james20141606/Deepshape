#! /usr/bin/env python
import numpy as np
import sys

if len(sys.argv) != 3:
    print >>sys.stderr, 'Usage: %s low_cutoff high_cutoff'
    sys.exit(1)

q = (float(sys.argv[1]), float(sys.argv[2]))
a = np.loadtxt(sys.stdin)
pct = np.percentile(a, q=q)
print '%f%%\t%f'%(q[0], pct[0])
print '%f%%\t%f'%(q[1], pct[1])
