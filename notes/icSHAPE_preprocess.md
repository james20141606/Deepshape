## Generate transcriptome FASTA files

### For human (on IBM-E)
1. Download reference genome sequences
```bash
cd ~/data/genomes/Human
wget ftp://ftp.ensembl.org/pub/release-88/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa
```
Download the GTF file:
```bash
wget ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz
```
Convert GTF to BED12 format for transcript:
```bash
cd ~/data/gtf/gencode.v26/
~/projects/tests/python/read_gtf.py \
    -i gencode.v26.annotation.gtf \
    --format bed12 --feature exon \
    -o transcript.bed12.bed
```
Generate FASTA file (hg38):
```bash
cd ~/data/genomes/fasta/Human/
bedtools getfasta -split -s -name -fi hg38.fa \
    -bed ~/data/gtf/gencode.v26/transcript.bed12.bed \
    | awk '/^>/{split($0, a, "::"); print a[1];next} {print toupper($0)}' \
    > hg38.transcript.v26.fa
awk '/^>/{split($0,a,".");print a[1];next}{print}' hg38.transcript.v26.fa \
    > hg38.transcript.v26.noversion.fa
samtools faidx hg38.transcript.v26.fa
samtools faidx hg38.transcript.v26.noversion.fa
```

convert GTF to BED12 format
```bash
cd ~/data/gtf/gencode.v19/
gtf_file=gencode.v19.annotation.gtf
~/projects/tests/python/read_gtf.py -i $gtf_file \
    --format bed12 --feature exon -o transcript.bed12.bed
```

Generate FASTA file (hg19):
```bash
cd data/genomes/fasta/Human/
bedtools getfasta -split -s -name -fi hg19.fa \
    -bed ~/data/gtf/gencode.v19/transcript.exonlist.bed \
    | awk '/^>/{split($0, a, "::"); print a[1];next} {print toupper($0)}' \
    > hg19.transcript.v19.fa
awk '/^>/{split($0,a,".");print a[1];next}{print}' hg19.transcript.v19.fa \
    > hg19.transcript.v19.noversion.fa
samtools faidx hg19.transcript.v19.fa
samtools faidx hg19.transcript.v19.noversion.fa
```
**Generate FASTA file using gffread (from samtools)**

```bash
gffread -g hg19.fa -s /Share/home/shibinbin/data/human.hg19.genome -W -M -F -G -A -O -E \
    -w hg19.transcript.v19.fa -d hg19.transcript.v19.collapsed.info \
    /Share/home/shibinbin/data/gtf/gencode.v19/gencode.v19.annotation.sorted.gtf
```

Generate bowtie2 index:
```bash
export PATH="/Share/home/shibinbin/pkgs/bowtie2/2.3.0/bin":$PATH
cd /Share/home/shibinbin/data/genomes/fasta/Human/
rm -f hg19.transcript.v19.*.bt2
bowtie2-build hg19.transcript.v19.fa hg19.transcript.v19
```

### For mouse (on IBM-E)
Download reference genome:
```bash
wget ftp://ftp.ensembl.org/pub/release-88/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz
```
Download the GTF file:
```bash
wget ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_mouse/release_M13/gencode.vM13.annotation.gtf.gz
```

```bash
cd data/gtf/gencode.vM12/
~/projects/tests/python/read_gtf.py \
    -i gencode.vM12.annotation.sorted.gtf \
    --format bed12 --feature exon \
    -o transcript.exonlist.bed
```
Generate FASTA file:
```bash
cd data/genomes/fasta/Mouse/
bedtools getfasta -split -s -name -fi mm10.fa \
    -bed ~/data/gtf/gencode.vM12/transcript.exonlist.bed \
    | awk '/^>/{split($0, a, "::"); print a[1];next} {print toupper($0)}' \
    > mm10.transcript.vM12.fa
awk '/^>/{split($0,a,".");print a[1];next}{print}' mm10.transcript.vM12.fa \
    > mm10.transcript.vM12.noversion.fa
samtools faidx mm10.transcript.vM12.fa
samtools faidx mm10.transcript.vM12.noversion.fa
```
Generate bowtie2 index:
```bash
export PATH="/Share/home/shibinbin/pkgs/bowtie2/2.3.0/bin":$PATH
cd /Share/home/shibinbin/data/genomes/fasta/Mouse/
rm -f mm10.transcript.vM12.*.bt2
bowtie2-build mm10.transcript.vM12.fa mm10.transcript.vM12
```

## Run icSHAPE pipeline
The icSHAPE pipeline can be downloaded from: (https://github.com/qczhang/icSHAPE).

1. Prepare a configuration file
Lines need to change in the sample configuration file:
```
JAVABIN         /usr/java/latest/bin/java
ADAPTER         $ICSHAPE/data/TruSeq2-PE.fa
ALIGNER         /srv/gs1/software/bowtie/2.0.5/bowtie2
MAPPINGREF      $ICSHAPE/data/bowtieLib/nanog
BEDGRAPH2BIGWIG /srv/gs1/software/ucsc_tools/2.7.2/bin/x86_64/bedGraphToBigWig
GENOMESIZE      /home/qczhang/database/ensembl/current/mouse/dna/genome.sm.chr.size
```

## Rerun mapping and following steps:
```bash
cd Lu_2016
rm -f *.sam *.sam.done *.rpkm *.rpkm.done *.rt *.rt.done icshape.* background.* target.*
cd ..
for data_name in Lu_2016_invitro Lu_2016_invivo;do
    rm -rf $data_name/
    mkdir $data_name/
    for f in $(ls Lu_2016);do
        ln -s ../Lu_2016/$f $data_name/
    done
done
```

## Convert icSHAPE files to GenomicData format
**For Human (hg19, v19)**, set:
```bash
gencode_version=v19
data_names="Lu_2016_invitro Lu_2016_invivo"
```
**For Human (published, hg19, v19)**, set:
```bash
gencode_version=v19
data_names="Lu_2016_invitro_published Lu_2016_invivo_published"
```
**For Human (hg38, v26)**, set:
```bash
gencode_version=v26
data_names="Lu_2016_invivo_hg38 Lu_2016_invitro_hg38"
```
**For Mouse (mm10, vM12)**, set:
```bash
gencode_version=vM12
data_names="Spitale_2015_invivo Spitale_2015_invitro"
```

**Then run**
```bash
for data_name in $data_names;do
    bin/preprocess.py IcshapeToGenomicData \
        -i output/icSHAPE_preprocess/${data_name}/icshape.out \
        -o data/icSHAPE/${data_name}/all.h5
    # create subsets of the data by genomic regions
    for region in CDS 5UTR 3UTR lncRNA miRNA ncRNA;do
        bin/preprocess.py SubsetGenomicDataByRegion \
            -i data/icSHAPE/${data_name}/all.h5 \
            --region-file $HOME/data/gtf/gencode.${gencode_version}/${region}.transcript.bed \
            --feature icshape \
            -o data/icSHAPE/${data_name}/${region}.h5
    done
done
```

## Plot the distribution of icSHAPE values
```bash
data_names=$(ls data/icSHAPE)
for data_name in $data_names;do
    bin/report.py GenomicDataDistribution \
        --infile data/icSHAPE/${data_name}/all.h5 \
        --feature icshape \
        --weight 1e-6 \
        --ylabel 'Counts (x1000000)' \
        --xlabel 'icSHAPE scores' \
        --outfile "reports/StructureScoreDistribution/icSHAPE/${data_name}/all.pdf"
done
```

## Summarize the the number of samples in the datasets
```bash
data_names=$(ls data/icSHAPE)
bin/report.py DeepfoldSampleSizeTable \
    --indir $(echo data/icSHAPE/*) \
    --outfile "reports/DeepfoldSampleSizeTable/icSHAPE/all.txt"
```

## Summarize the number of samples in the datasets by region
```bash
bin/report.py SampleSizeTable \
    --indir data/icSHAPE \
    --outfile "reports/SampleSizeTable/icSHAPE.txt"
```

## Plot base distribution (stacked bar)
```bash
for p in 5;do
    for region in all;do
        for data_name in $data_names;do
            bin/preprocess.py BaseDistribution \
                --feature icshape \
                --percentile ${p} \
                --plot-type stacked_bar \
                --score-file data/icSHAPE/${data_name}/${region}.h5 \
                --sequence-file /Share/home/shibinbin/data/gtf/gencode.${gencode_version}/sequences/${region}.transcript.fa \
                --outfile "reports/BaseDistribution/icSHAPE/${data_name}/r=${region}.p=${p}.stacked_bar.pdf"
        done
    done
done
```

## Plot logistic regression weights
```bash
for data_name in $data_names;do
    bin/report.py LogRegWeights \
        --infile "trained_models/icSHAPE/${data_name}/r=all,p=5,w=75,m=logreg.h5" \
        --outfile "reports/LogRegWeights/icSHAPE/${data_name}/r=all,p=5,w=75,m=logreg.pdf"
done
```

## Compare ROC curves of different methods
```bash
for data_name in $data_names;do
    bin/report.py RocCurve \
        --region all --bykey model --plot-type roc_curve \
        --indir "metrics/icSHAPE/$data_name" \
        --title "Classification performance ($data_name)" \
        --outfile "reports/RocCurveByModel/$data_name/r=all.pdf"
done
```

## Compare ROC curves of different window size
```bash
for data_name in $data_names;do
    bin/report.py RocCurve \
        --region all --bykey model --xkey window_size --plot-type auc_lines \
        --indir "metrics/icSHAPE/$data_name" \
        --title "$data_name" \
        --outfile "reports/AucAcrossWindowSizeByModel/$data_name/r=all.pdf"
done
```

## Plot base distribution
First set environment variables `$gencode_version` and `$data_names` same as the previous section.

```bash
for p in 5;do
    for region in CDS 5UTR 3UTR lncRNA all;do
        for data_name in $data_names;do
            bin/preprocess.py BaseDistribution \
                --feature icshape \
                --percentile ${p} \
                --score-file data/icSHAPE/${data_name}/${region}.h5 \
                --sequence-file /Share/home/shibinbin/data/gtf/gencode.${gencode_version}/sequences/${region}.transcript.fa \
                --outfile "reports/BaseDistribution/icSHAPE/${data_name}/r=${region}.p=${p}.pdf"
        done
    done
done
```

## Predict the icSHAPE values on known structure
```bash
for data_name in $data_names;do
    bin/deepfold2.py PredictDeepfold1D \
        --swap-labels \
        --split --format fasta --fillna 0.5 \
        --infile data/Known/All/sequences.fa \
        --model-file "trained_models/icSHAPE/$data_name/r=all,p=5,w=50,m=logreg.h5" \
        --outfile "output/Known/icSHAPE,${data_name}/r=all,p=5,w=50,m=logreg"
done
```

## Draw known structure annotated with predicted values
```bash
for data_name in $data_names;do
    for seqname in $(cat data/Known/selected_sequences.txt);do
    bin/report.py DrawRnaStructureWithValues \
        --varna-path $HOME/apps/archive/VARNAv3-93.jar \
        --ct-file data/Known/ct/${seqname}.ct \
        --value-file "output/Known/icSHAPE,${data_name}/r=all,p=5,w=50,m=logreg/$seqname" \
        --outfile "reports/DrawRnaStructureWithValues/icSHAPE/${data_name}/r=all,p=5,w=50,m=logreg/${seqname}.svg"
    done
done
```
## Get transcript-based coordinates by region:

**For Human (hg19 v19)**, set environment variables:
```bash
gencode_version=v19
genome_version=hg19
species=Human
```
**For Human (hg38 v26)**, set environment variables:
```bash
gencode_version=v26
genome_version=hg38
species=Human
```

**For Mouse (mm10, vM12)**, set environment variables:
```bash
gencode_version=vM12
genome_version=mm10
species=Mouse
```

```bash
for region in CDS 5UTR 3UTR;do
    ~/projects/tests/python/read_gtf.py -i gencode.${gencode_version}.annotation.gtf \
        --format transcript_bed --feature ${region} -o ${region}.transcript.bed
done
~/projects/tests/python/read_gtf.py -i gencode.${gencode_version}.long_noncoding_RNAs.gtf \
    --format transcript_bed --feature transcript -o lncRNA.transcript.bed
~/projects/tests/python/read_gtf.py -i gencode.${gencode_version}.annotation.gtf \
    --transcript-types miRNA \
    --format transcript_bed --feature transcript -o miRNA.transcript.bed
~/projects/tests/python/read_gtf.py -i gencode.${gencode_version}.annotation.gtf \
    --transcript-types rRNA,snRNA,snoRNA,scaRNA,scRNA,Mt_rRNA,Mt_tRNA,ribozyme,misc_RNA \
    --format transcript_bed --feature transcript -o ncRNA.transcript.bed
for region in CDS 5UTR 3UTR lncRNA miRNA ncRNA;do
    if awk '{ if(index($4, "-") == 0) {exit 1} else {exit 0}}' ${region}.transcript.bed;then
        continue
    fi
    awk -v r=$region 'BEGIN{OFS="\t"}{$4=$4 "-" r; print}' ${region}.transcript.bed  > ${region}.transcript.bed.renamed
    mv ${region}.transcript.bed.renamed ${region}.transcript.bed
done
```
Get FASTA sequences
```bash
[ -d sequences ] || mkdir sequences
for region in CDS 5UTR 3UTR lncRNA ncRNA miRNA;do
    ~/projects/tests/python/get_fasta.py -name \
        -fi ~/data/genomes/fasta/${species}/${genome_version}.transcript.${gencode_version}.fa \
        -format '{name}' \
        -s -bed ${region}.transcript.bed > sequences/${region}.transcript.fa
    samtools faidx sequences/${region}.transcript.fa
done
```

## Convert RT-stop counts to GenomicData format
```bash
data_names="Lu_2016_invitro Lu_2016_invivo Spitale_2015_invitro Spitale_2015_invivo"
for data_name in $data_names;do
    bin/preprocess.py IcshapeRtToGenomicData -i output/icSHAPE_preprocess/$data_name/background.normalized.rt \
        --normalized -o data/icSHAPE/$data_name/raw/background_normalized
    bin/preprocess.py IcshapeRtToGenomicData -i output/icSHAPE_preprocess/$data_name/target.normalized.rt \
        --normalized -o data/icSHAPE/$data_name/raw/target_normalized
    bin/preprocess.py IcshapeRtToGenomicData -i output/icSHAPE_preprocess/$data_name/background.rt \
        -o data/icSHAPE/$data_name/raw/background
    bin/preprocess.py IcshapeRtToGenomicData -i output/icSHAPE_preprocess/$data_name/target.rt \
        -o data/icSHAPE/$data_name/raw/target
done
```
## Calculate structures for icSHAPE
```bash
data_name=Spitale_2015_invitro
region=all
gencode_version=vM12
window_size=128
for score_method in count_diff icshape_nowinsor background target;do
    bin/preprocess.py CalculateStructureScoresForIcshape \
        --background-file data/icSHAPE/$data_name/raw/background_normalized \
        --target-file data/icSHAPE/$data_name/raw/target_normalized \
        --rpkm-cutoff 30 \
        --method $score_method \
        -o data/icSHAPE/$data_name/$score_method/all
done
```
## Get base distribution of bins between percentiles of calculated icSHAPE scores
```bash
score_method=icshape
score_method=icshape_nowinsor
score_method=count_diff
score_method=background
score_method=target
if [ "$score_method" = icshape ];then
    input_file=data/icSHAPE/$data_name/all
    feature=icshape
else
    input_file=data/icSHAPE/$data_name/$score_method/all
    feature=$score_method
fi
for bin_method in value percentile;do
    for offset in $(seq -3 3);do
        bin/report.py BaseDistributionByBin \
            -i $input_file \
            --sequence-file data/gtf/gencode.${gencode_version}/sequences/${region}.transcript.fa \
            --bins 20 --bin-method $bin_method --offset $offset \
            -o reports/BaseDistributionByBin/icSHAPE/${data_name}/$score_method/${bin_method}.${offset}
    done
done
```
## Calculate base-pair 1D profiles using RNAplfold algorithm
```bash
gencode_version=vM12
region=all
data_name=Spitale_2015_invitro
window_size=100
bin/preprocess.py RNAfoldBasePairProfile -i data/gtf/gencode.${gencode_version}/sequences/${region}.transcript.fa \
    --window-size $window_size -j 10 --batch-size 50 \
    -o data/RNAfoldBasePairProfile/gencode.${gencode_version}/${region}.${window_size} \
    --names-file data/icSHAPE/$data_name/all:name
```

```bash
bin/preprocess.py ExtractBigWig \
    --plus-bigwig-file /Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq/DMS/genome.plus.bw \
    --minus-bigwig-file /Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq/DMS/genome.minus.bw \
    --bed-file /Share/home/zhuyumin/Genome/Arabidopsis/TAIR/TAIR10_GFF3_genes.bed12 \
    --feature-name dmsseq \
    -o tmp/DMSSeq_TAIR10/DMS
bin/preprocess.py ExtractBigWig \
    --plus-bigwig-file /Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq/Control/genome.plus.bw \
    --minus-bigwig-file /Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq/Control/genome.minus.bw \
    --bed-file /Share/home/zhuyumin/Genome/Arabidopsis/TAIR/TAIR10_GFF3_genes.bed12 \
    --feature-name dmsseq \
    -o tmp/DMSSeq_TAIR10/Control
bin/preprocess.py CalcDmsseqScores \
    --treatment-file tmp/DMSSeq_TAIR10/DMS \
    --control-file tmp/DMSSeq_TAIR10/Control \
    -o tmp/DMSSeq_TAIR10/scores
```
```python
filename = 'tmp/DMSSeq_TAIR10/scores'
output_file = 'tmp/DMSSeq_TAIR10/scores_by_transcript_id'

import h5py
with h5py.File(filename, 'r') as f:
    names = f['name'][:]
    start = f['start'][:]
    end = f['end'][:]
    scores = f['/feature/dmsseq'][:]
with h5py.File(output_file, 'w') as f:
    for i in range(len(names)):
        f.create_dataset(names[i], data=scores[start[i]:end[i]])
```
