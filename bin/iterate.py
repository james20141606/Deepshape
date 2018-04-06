#! /usr/bin/env python
from itertools import product
import sys
delimiter = ' '
if len(sys.argv) < 2:
    sys.exit(0)
for items in product(*map(lambda x: x.split(delimiter), sys.argv[1:])):
    print '\t'.join(items)
