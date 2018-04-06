import ctypes
from ctypes.util import find_library
import sys, os
import numpy as np
import logging
logger = logging.getLogger('bigwig')

"""Steps to build the jkweb shared library:
* Download the source from http://hgdownload.soe.ucsc.edu/admin/exe/userApps.src.tgz
* Enter the kent/src directory.
* Make sure that openssl-devel, libpng-devel, zlib-devel are installed
* Modify lib/makefile to add the following lines:
%.o: %.c
        ${CC} -fPIC ${COPT} ${CFLAGS} ${HG_DEFS} ${LOWELAB_DEFS} ${HG_WARN} ${HG_INC} ${XINC} -o $@ -c $<

$(MACHTYPE)/libjkweb.so: $(O) $(MACHTYPE)
    gcc -fPIC -shared -o $(MACHTYPE)/libjkweb.so $(O) -Wl,-z,defs -L../htslib -lhts -lm -lz -lpng -lpthread -lcrypto -lssl
* Enter the htslib directory, run the following commands to generate libhts.a
CFLAGS="-fPIC -DUCSC_CRAM=0 -DNETFILE_HOOKS=0" ./configure
make
* Enter the lib/ directory. Run
make x86_64/libjkweb.so
"""

class bbiInterval(ctypes.Structure):
    '''inc/common.h:
#define bits32 unsigned
----------------------
inc/bbiFile.h:
struct bbiInterval
/* Data on a single interval. */
    {
    struct bbiInterval *next;   /* Next in list. */
    bits32 start, end;          /* Position in chromosome, half open. */
    double val;             /* Value at that position. */
    };
    '''
    pass

bbiInterval._fields_ = [('next', ctypes.POINTER(bbiInterval)),
            ('start', ctypes.c_uint),
            ('end', ctypes.c_uint),
            ('val', ctypes.c_double)
            ]

class BigWigFile(object):
    def __init__(self, bigwig_file):
        self.bwf = bigWigFileOpen(bigwig_file)
        if not self.bwf:
            raise IOError('cannot open the bigwig file: ' + bigwig_file)
        self.lm = ctypes.c_void_p(lmInit(0))

    def close(self):
        bbiFileClose(self.bwf)
        lmCleanup(ctypes.byref(self.lm))
        self.bwf = None

    def interval_query(self, chrom, start, end):
        """Returns a numpy array of all values in range chrom:start-end
        """
        interval = bigWigIntervalQuery(self.bwf, chrom, start, end, self.lm)
        values = np.full(end - start, np.nan, dtype='float64')
        while interval:
            interval = interval.contents
            values[(interval.start - start) : (interval.end - end)] = interval.val
            interval = interval.next
        return values

    def interval_query_blocked(self, chrom, start, end, block_starts, block_sizes):
        """Similar to interval_query except that only values in the blocks are fetched
        """
        n_blocks = len(block_starts)
        value_starts = np.cumsum([0] + block_sizes)[:-1]
        length = np.sum(block_sizes)
        values = np.full(length, np.nan, dtype='float64')

        for i, block_start, block_size in zip(range(n_blocks), block_starts, block_sizes):
            block_start += start
            interval = bigWigIntervalQuery(self.bwf, chrom,
                block_start, block_start + block_size, self.lm)
            while interval:
                interval = interval.contents
                values[(interval.start - block_start + value_starts[i]) :
                    (interval.end - block_start + value_starts[i])] = interval.val
                interval = interval.next
        return values

    def __del__(self):
        if self.bwf is not None:
            self.close()

def init_jkweb(dll_path):
    '''inc/bigWig.h:
    struct bbiFile *bigWigFileOpen(char *fileName);
    /* Open up big wig file.   Free this up with bbiFileClose */
    #define bigWigFileClose(a) bbiFileClose(a)
    struct bbiInterval *bigWigIntervalQuery(struct bbiFile *bwf, char *chrom, bits32 start, bits32 end,
        struct lm *lm);
    ----------------------
    inc/bbiFile.h:
    void bbiFileClose(struct bbiFile **pBwf);
    /* Close down a big wig/big bed file. */
    ----------------------
    inc/localmem.h
    struct lm *lmInit(int blockSize);
    /* Create a local memory pool. Parameters are:
     *      blockSize - how much system memory to allocate at a time.  Can
     *                  pass in zero and a reasonable default will be used.
     */
    void lmCleanup(struct lm **pLm);
    /* Clean up a local memory pool. */
    '''
    # find the dynamic library named libjkweb.so
    jkweb = ctypes.cdll.LoadLibrary(dll_path)

    # import all functions in jkweb into the global namespace
    symbols = ('bigWigFileOpen', 'bbiFileClose', 'bigWigIntervalQuery',
        'lmInit', 'lmCleanup')
    for sym in symbols:
        logger.debug('import symbol %s from %s'%(sym, dll_path))
        globals()[sym] = getattr(jkweb, sym)

    # set argument types and result types for imported functions
    bigWigFileOpen.argtypes = (ctypes.c_char_p,)
    bigWigFileOpen.restype = ctypes.c_void_p
    bbiFileClose.argtypes = (ctypes.c_void_p,)
    bbiFileClose.restype = None
    bigWigIntervalQuery.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)
    bigWigIntervalQuery.restype = ctypes.POINTER(bbiInterval)
    lmInit.argtypes = (ctypes.c_int,)
    lmInit.restype = ctypes.c_void_p
    lmCleanup.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    lmCleanup.restype = None
