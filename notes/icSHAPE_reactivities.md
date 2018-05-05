## Get FastQ files
```bash
cd /lulab/lustre2/shibinbin/data
data_dir=/800T/luzhi/users/shibinbin/data/SRA
# Lu_2016
{
for sra_id in $(awk '{print $1}' $data_dir/icSHAPE-Lu-2016.txt);do
    echo fastq-dump -Z $data_dir/${sra_id}.sra '|' pigz -c '>' fastq/${sra_id}.fastq.gz
done
} | parallel -t -j6
# Spitale_2015
{
for sra_id in $(awk '{print $1}' $data_dir/icSHAPE-Spitale-2015.txt);do
    echo fastq-dump -Z $data_dir/${sra_id}.sra '|' pigz -c '>' fastq/${sra_id}.fastq.gz
done
} | parallel -t -j6
```

## Run mapping
```bash
output_dir=output/icSHAPE/mapping
{
for sample in $(awk '{print $1}' metadata/icSHAPE-Lu-2016.txt);do
    [ -d "$output_dir/${sample}.bowtie" ] || mkdir -p "$output_dir/${sample}.bowtie"
    echo pigz -d -c output/icSHAPE/reads/${sample}.trimmed.fastq.gz \
        '|' bowtie2 -U /dev/stdin -S /dev/stdout \
        -x output/transcriptomes/human_known_rnas.bowtie2_index/transcriptome \
        -p 8 --non-deterministic --time \
        '|' samtools view -b '>' $output_dir/${sample}.bowtie/Aligned.out.bam
done
} | parallel -t -j 6
```
```
for dataset in Lu_2016_DMSO Lu_2016_invitro Lu_2016_invivo;do
    bin/preprocess.py IcshapeRtToGenomicData \
        -i output/icSHAPE/reactivities/${dataset}.human_known_rnas/target.normalized.rt \
        --normalized -o output/icSHAPE/reactivities/${dataset}.human_known_rnas/target.normalized.rt.h5
    bin/preprocess.py IcshapeRtToGenomicData \
        -i output/icSHAPE/reactivities/${dataset}.human_known_rnas/target.rt \
        -o output/icSHAPE/reactivities/${dataset}.human_known_rnas/target.rt.h5
done
```