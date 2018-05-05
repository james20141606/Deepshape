## Extract counts from bigwig files
```bash
/Share/home/shibinbin/projects/tests/python/read_gtf.py
awk 'BEGIN{c["1"]="Chr1"; c["2"]="Chr2"; c["3"]="Chr3"; c["4"]="Chr4"; c["5"]="Chr5";
    c["Mt"]="mitochondria"; c["Pt"]="chloroplast"; FS="\t"; OFS="\t"} {$1=c[$1]; print}' \
    /Share/home/shibinbin/data/gtf/TAIR10/transcript.bed12.bed > /Share/home/shibinbin/data/gtf/TAIR10/transcript.bed12.bed.1
mv /Share/home/shibinbin/data/gtf/TAIR10/transcript.bed12.bed.1 /Share/home/shibinbin/data/gtf/TAIR10/transcript.bed12.bed
data_dir=/Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq
[ -d "data/Structure-seq" ] || mkdir -p "data/Structure-seq"
for data_name in Control DMS;do
    bin/preprocess.py ExtractBigWig \
        --plus-bigwig-file $data_dir/$data_name/genome.plus.bw \
        --minus-bigwig-file $data_dir/$data_name/genome.minus.bw \
        --bed-file /Share/home/shibinbin/data/gtf/TAIR10/transcript.bed12.bed \
        -o data/Structure-seq/A.thaliana/$data_name.counts
done
```
/Share/home/shibinbin/projects/Deepfold2/data/Structure-seq/A.thaliana/Control.counts
/Share/home/shibinbin/projects/Deepfold2/data/Structure-seq/A.thaliana/DMS.counts
```R
library(rhdf5)
filename <- '/Share/home/shibinbin/projects/Deepfold2/data/Structure-seq/A.thaliana/DMS.counts'
start <- h5read(filename, '/start')
start <- start + 1
end <- h5read(filename, '/end')
counts <- h5read(filename, '/feature/data'))
name_to_index <- as.list(1:length(name))
names(name_to_index) <- h5read(filename, '/name')
# get counts for one transcript
i <- name_to_index[['ATCG01310.1']]
head(counts[start[i]:end[i]], n=100)
```

## Calculate DMS-seq reactivities
```bash
data_dir=/Share/home/shibinbin/data/RNAex/count/A.thaliana/TAIR10/Structure-seq
output_dir=output/DMS-seq/Ding_2014
[ -d "$output_dir" ] || mkdir -p "$output_dir"
gffread --bed -o  - data/gtf/TAIR10/Arabidopsis_thaliana.TAIR10.34.gtf \
    | awk 'BEGIN{OFS="\t"}{if($1 ~ /^[0-9]+/){$1="Chr"$1}
    else if($1 == "Mt") {$1="mitochondria"} 
    else if($1=="Pt") {$1="chloroplast"}
    else {next} 
    print}' \
    > data/gtf/TAIR10/Arabidopsis_thaliana.TAIR10.34.bed12
bin/calculate_reactivity.py dms_seq \
    --treatment-plus-file $data_dir/DMS/genome.plus.bw \
    --treatment-minus-file $data_dir/DMS/genome.minus.bw \
    --control-plus-file $data_dir/Control/genome.plus.bw \
    --control-minus-file $data_dir/Control/genome.minus.bw \
    --bed-file data/gtf/TAIR10/Arabidopsis_thaliana.TAIR10.34.bed12 \
    -o $output_dir/reactivities
```