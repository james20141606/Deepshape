#! /usr/bin/env python
import sys, os, argparse
import re
from collections import defaultdict

def get_geneid_tsid(f):
    pat = re.compile(r'(?P<key>gene_id|transcript_id) "(?P<value>[^"]+)";')
    transcript_count = defaultdict(int)
    tsid_to_geneid = {}
    for lineno, line in enumerate(f):
        if line.startswith('#'):
            continue
        fields = line.split('\t')
        if fields[2] == 'transcript':
            m = pat.findall(fields[8])
            if m is not None:
                d = dict(m)
                #print '\t'.join((d['gene_id'], d['transcript_id']))
                d['transcript_id'] = d['transcript_id'].split('.')[0]
                transcript_count[d['gene_id']] += 1
                tsid_to_geneid[d['transcript_id']] = d['gene_id']
    #for tsid, count in transcript_count.iteritems():
    #    print '{}\t{}'.format(tsid, count)
    for tsid, geneid in tsid_to_geneid.iteritems():
        print '{}\t{}'.format(tsid, transcript_count[geneid])

if __name__ == '__main__':
    get_geneid_tsid(sys.stdin)
