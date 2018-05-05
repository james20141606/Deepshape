# Analysis of published icSHAPE data (Lu et al. 2016) (GSE74353)

## Preprocessing

1. Find deprecated ensembl IDs in the published icSHAPE data
```bash
cd output/icSHAPE_preprocess/GSE74353
awk '{print $0,NF-3}' GSE74353_HS_293T_icSHAPE_InVitro_BaseReactivities.txt > invitro.length.txt
awk '{print $0,NF-3}' GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt > invivo.length.txt
awk '{split($1,a,".");print a[1],$2}' ~/data/genomes/fasta/Human/hg19.transcript.v19.fa.fai > hg19.transcript.v19.length.txt
awk 'FNR==NR{a[$1]=$2;next}{b[$1]=$2}END{for(i in a){if(a[i]!=b[i]) print i,a[i],b[i]}}' \
    invitro.length.txt hg19.transcript.v19.length.txt > deprecated_ids.invitro.txt
awk 'FNR==NR{a[$1]=$2;next}{b[$1]=$2}END{for(i in a){if(a[i]!=b[i]) print i,a[i],b[i]}}' \
    invivo.length.txt hg19.transcript.v19.length.txt > deprecated_ids.invivo.txt
cat deprecated_ids.invitro.txt deprecated_ids.invivo.txt | sort | uniq > deprecated_ids.txt
```
2. List of deprecated ensembl IDs:
```
ENST00000374449 3060
ENST00000400776 2370
ENST00000474885 153
ENST00000488948 114
ENST00000581141 92
ENST00000584459 180
ENST00000606783 1869
ENST00000607521 5070
```
3. Remove deprecated ensembl IDs:
```bash
awk 'BEGIN{OFS="\t"}FNR==NR{a[$1]=1;next}{if(!($1 in a)) print}' \
    deprecated_ids.txt GSE74353_HS_293T_icSHAPE_InVitro_BaseReactivities.txt \
    > invitro.icshape.txt
awk 'BEGIN{OFS="\t"}FNR==NR{a[$1]=1;next}{if(!($1 in a)) print}' \
    deprecated_ids.txt GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt \
    > invivo.icshape.txt
```
4. Append transcript version number from GENCODE v19
```bash
for dataset in invitro invivo;do
awk 'BEGIN{OFS="\t"} FNR==NR{split($1,a,".");tid[a[1]]=$1;next} {if($1 in tid){$1=tid[$1];print}}' \
    ~/data/genomes/fasta/Human/hg19.transcript.v19.fa.fai \
    ${dataset}.icshape.txt > ${dataset}.icshape.v19.txt
    [ -d ../Lu_2016_${dataset}_published ] || mkdir ../Lu_2016_${dataset}_published
    cp ${dataset}.icshape.v19.txt ../Lu_2016_${dataset}_published/icshape.out
done
```
4. Convert icSHAPE files to GenomicData format
```bash
for dataset in invitro invivo;do
    bin/preprocess.py IcshapeToGenomicData \
        -i output/icSHAPE_preprocess/Lu_2016_${dataset}_published/icshape.out \
        -o data/icSHAPE/Lu_2016_invitro_published/all.h5
done
```
5. Strip the version number from the sequence names
```bash
cd ~/data/genomes/fasta/Human
awk '/^>/{split($0,a,".");print a[1];next}{print}' hg19.transcript.v19.fa > hg19.transcript.v19.noversion.fa
samtools faidx hg19.transcript.v19.noversion.fa
```
5. Create datasets for Deepfold

Add a task in create_jobs.py
```python
class CreateDatasetForIcshapeHumanPublished(Task):
    def build(self):
        self.paramlist = ParamGrid({
            'data_name': ['Lu_2016_invitro_published', 'Lu_2016_invivo_published'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'sequence_file': ['~/data/genomes/fasta/Human/hg19.transcript.v19.noversion.fa']
        }).to_list()
        self.tool = CreateDatasetForIcshape()
```
Then run the job:
```bash
rm -rf data/icSHAPE/Lu_2016_invivo_published/deepfold
rm -rf data/icSHAPE/Lu_2016_invitro_published/deepfold
rm -rf CreateDatasetForIcshapeHumanPublished
bin/create_jobs.py CreateBsubJob -t CreateDatasetForIcshapeHumanPublished
bsub < jobs/CreateDatasetForIcshapeHumanPublished.sh
```

## Insect the data

### Plot base distribution
```bash
for p in 5 10 20 30 40;do
    for data_name in Lu_2016_invitro_published Lu_2016_invivo_published;do
        bin/preprocess.py BaseDistribution \
            --feature icshape \
            --percentile ${p} \
            --score-file data/icSHAPE/${data_name}/icshape.h5 \
            --sequence-file /Share/home/shibinbin/data/genomes/fasta/Human/hg19.transcript.v19.noversion.fa \
            --outfile "reports/BaseDistribution/icSHAPE/${data_name}.p=${p}.pdf"
    done
done
```


## Train Deepfold model
1. Create task files in `bin/create_jobs.sh`:
```python
class TrainDeepfold1DForIcshapeLogreg(Task):
    def build(cls):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'model_name': ['logreg']
        }).to_list()
        self.tool = TrainDeepfold1DForIcshape()

class TrainDeepfold1DForIcshapeCnn(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'percentile': [5],
            'window_size': [75, 100, 125, 150, 175, 200],
            'model_name': ['conv1', 'fcn1', 'fcn2']
        }).to_list()
        self.tool = TrainDeepfold1DForIcshape()

class TrainDeepfold1DForIcshapeBlstm(Task):
     def build(self):
         self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
             'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
             'Spitale_2015_invitro', 'Spitale_2015_invivo'],
             'percentile': [5],
             'window_size': [25, 50, 75, 100],
             'model_name': ['blstm3']
         }).to_list()
         self.tool = TrainDeepfold1DForIcshape()
```
Then submit the jobs:
```bash
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeLogreg
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeCnn
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeBlstm
```
2. Check the status of the jobs:
```bash
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeLogreg
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeCnn
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeBlstm
```
## Evaluate the models
1. Evalualte the models

Create tasks in `bin/create_jobs.py`:
```python
class EvaluateDeepfold1DForIcshapeLogReg(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'model_name': ['logreg']
        }).to_list()
        self.tool = EvaluateDeepfold1DForIcshape()

class EvaluateDeepfold1DForIcshapeCnn(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'percentile': [5],
            'window_size': [75, 100, 125, 150, 175, 200],
            'model_name': ['conv1', 'fcn1', 'fcn2']
        }).to_list()
        self.tool = EvaluateDeepfold1DForIcshape()

class EvaluateDeepfold1DForIcshapeBlstm(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100],
            'model_name': ['blstm3']
        }).to_list()
        self.tool = EvaluateDeepfold1DForIcshape()
```
2. Submit the jobs
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeLogReg
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeCnn
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeBlstm
```
3. Check the status
```bash
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeLogReg
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeCnn
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeBlstm
```
## Report metrics and select models
1. Report metrics:
```bash
for data_name in Lu_2016_invitro_published Lu_2016_invivo_published;do
    bin/report.py MetricTable --experiment-type icSHAPE --data-name $data_name \
        -o reports/MetricTable/icSHAPE/${data_name}.txt
    bin/report.py SelectModel --metric accuracy --metric-file reports/MetricTable/icSHAPE/${data_name}.txt \
        --num 3 -o selected_models/icSHAPE/${data_name}.json
done
```

## Compare icSHAPE values (with published data)
```bash
for data_name in Lu_2016_invitro Lu_2016_invivo;do
bin/preprocess.py CorrelationBetweenIcshape \
    --strip-transcript-version \
    --transcript-anno ~/data/gtf/gencode.v19/transcript.bed \
    --name1 'Processed' --infile1 data/icSHAPE/${data_name}/icshape.h5 \
    --name2 'Published' --infile2 data/icSHAPE/${data_name}_published/icshape.h5 \
    --prefix reports/CorrelationBetweenIcshapeAndPublished/${data_name} \
    --title "Correlation between icSHAPE data and published data (${data_name})"
done
```


## Copy to shared directory (BioII)
```bash
output_dir=/BioII/lulab_b/shared/projects/icSHAPE
datasets="Lu_2016_invitro_hg38 Lu_2016_invivo_hg38 Lu_2016_invitro_hg19 Lu_2016_invivo_hg19 Spitale_2015_invitro_mm10 Spitale_2015_invivo_mm10"
set -x
for dataset in $datasets;do
    mkdir -p $output_dir/$dataset
    rsync -rav --dry-run output/icSHAPE_preprocess/$dataset/{icshape.out,icshape.tmp.out,target.rt,target.normalized.rt} \
        output/icSHAPE_preprocess/$dataset/{background.rt,background.normalized.rt} \
        output/icSHAPE_preprocess/$dataset/*.h5 \
        $output_dir/$dataset
done
mv $output_dir/Lu_2016_invitro $output_dir/Lu_2016_invitro_hg19
mv $output_dir/Lu_2016_invivo $output_dir/Lu_2016_invivo_hg19
mv $output_dir/Spitale_2015_invitro $output_dir/Spitale_2015_invitro_mm10
mv $output_dir/Spitale_2015_invivo $output_dir/Spitale_2015_invivo_mm10
set +x
```

## Convert icshape.out to HDF5 format
```bash
{
for dataset in $datasets;do
    bin/icshape.py icshape_to_hdf5 \
        -i output/icSHAPE_preprocess/$dataset/icshape.out \
        -o output/icSHAPE_preprocess/$dataset/icshape.h5
done
} | parallel -j4
{
for dataset in $datasets;do
    for library in background target;do
        bin/icshape.py rt_to_hdf5 \
            -i output/icSHAPE_preprocess/$dataset/${library}.rt \
            --rt-file output/icSHAPE_preprocess/$dataset/${library}.rt.h5 \
            --bd-file output/icSHAPE_preprocess/$dataset/${library}.bd.h5
    done
done
}
```


```bash
export ICSHAPE=$PWD/tools/icSHAPE
for dataset in Lu_2016_invitro;do
    perl $ICSHAPE/scripts/enrich2Bedgraph.pl \
        -i output/icSHAPE_preprocess/$dataset/icshape.out \
        -g data/gtf/gencode.v19/gencode.v19.annotation.gtf \
        -a data/genomes/fasta/Human/hg19.transcript.v19.fa \
        -o /dev/stdout \
    | sort -k1,1 -k2,3n > output/icSHAPE_preprocess/$dataset/icshape.bedGraph
    perl $ICSHAPE/scripts/uniqueTrack.pl \
        -i output/icSHAPE_preprocess/$dataset/icshape.bedGraph \
        -o /dev/stdout \
    | sort -f1-4 | grep -v NULL > output/icSHAPE_preprocess/$dataset/icshape.sim.bedGraph
    bedGraphToBigWig icSHAPE.sim.bedGraph data/chrom_sizes/hg19 output/icSHAPE_preprocess/$dataset/icshape.bigWig
    
done
```
