## Convert RDAT files to Python pickle format
```python
import os, cPickle, gzip
import rdatkit
rmdb_dir = '/Share/home/shibinbin/data/RMDB'
rdat_files = {}
for filename in os.listdir(rmdb_dir):
    rdat = rdatkit.RDATFile()
    with open(os.path.join(rmdb_dir, filename), 'r') as f:
        rdat.load(f)
    rdat_files[filename] = rdat
with gzip.open('data/RMDB.pkl.gz', 'wb') as f:
    cPickle.dump(rdat_files, f, protocol=cPickle.HIGHEST_PROTOCOL)
```
## Load the RDAT pickle file
```python
import gzip, cPickle
with gzip.open('data/RMDB.pkl.gz', 'rb') as f:
    rdat_files = cPickle.load(f)
```

## Get IDs of mutate-and-map experiments
```bash
{
for f in ~/data/RMDB/*.rdat;do
    if [ -n "$(grep experimentType:MutateAndMap $f)" ];then
        basename $f
    fi
done
} | cut -d'.' -f1 > data/RMDB/MutateAndMap.ID.txt
```
## Extract sequences (wild type) from RDAT files
```bash
bin/preprocess.py ExtractRdat \
    -i data/RMDB/MutateAndMap.ID.txt \
    --rmdb-dir /Share/home/shibinbin/data/RMDB \
    --format rnafold \
    -o data/RMDB/MutateAndMap.rnafold.txt
bin/deepfold2.py PredictDeepfold1D \
    -i data/RMDB/MutateAndMap.rnafold.txt \
    --format rnafold \
    --swap-labels \
    --model-file 'trained_models/icSHAPE/Lu_2016_invitro/r=CDS,p=5,w=100,m=logreg' \
    --metric-by-sequence-file /dev/stdout
```
Removed `X20H20_DMS_0001` and `X20H20_DMS_0002` because no mutation annotation was found.
Removed `RNASEP_DMS_0000`, `TRP4P6_DMS_0005`, `TRP4P6_DMS_0006` because the each base is mutated to 'X'.
## Predict reactivity
```bash
bin/preprocess.py ExtractRdat \
    -i data/RMDB/MutateAndMap.ID.txt \
    --rmdb-dir /Share/home/shibinbin/data/RMDB \
    --format genomic_data \
    -o data/RMDB/MutateAndMap.genomic_data
bin/deepfold2.py PredictDeepfold1D \
    -i data/RMDB/MutateAndMap.genomic_data \
    --format genomic_data \
    --model-file 'trained_models/icSHAPE/Lu_2016_invitro/r=CDS,p=5,w=100,m=logreg' \
    --metrics pearson_r,spearman_r \
    --metric-by-sequence-file tmp/MutateAndMap.reactivity.metric_by_sequence.txt
```
