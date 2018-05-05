## Data
### Download processed data
* Nature Methods 2014: (http://www.chem.unc.edu/rna/data-files/shape-map_DATA.zip)
* Cell 2018: (http://www.chem.unc.edu/rna/data-files/mustoe_2018_DATA_SOFTWARE.zip)

### Download raw data
**Cell 2018**

https://www.ebi.ac.uk/ena/data/view/PRJEB23974
```bash
{
for sra_id in `awk 'BEGIN{OFS="\t"}NR>1{print $5}' metadata/PRJEB23974.txt`;do
    echo fastq-dump -O fastq --gzip --split-files SRA/${sra_id}.sra ';' \
        pigz fastq/${sra_id}_*.fastq
done
} | parallel -j8 -t
```

## Data preprocessing (Cell 2018)

### Set environment variables
```bash
data_dir=data/SHAPE-MaP/Cell_2018/mustoe_2018_DATA_SOFTWARE/Mustoe2018_data
output_dir=output/SHAPE-MaP/Cell_2018
[ -d "$output_dir" ] || mkdir -p $output_dir
```

### Get fasta sequences
```bash
awk 'BEGIN{FS=",";OFS="\t"}NR>1{print "U00096.2",$3-1,$4,$1,0,$2}' \
    $data_dir/transcripts.txt > $output_dir/transcripts.bed
bedtools getfasta -s -fi $data_dir/U00096.2.fa -bed $data_dir/transcripts.bed -name \
    | awk '{gsub(/::.*/,""); print}' > $output_dir/transcripts.fa
awk 'BEGIN{FS=",";OFS="\t"}NR>1{print "U00096.2",$2-1,$3,$1,0,"+"}' \
    $data_dir/genes.txt > $output_dir/orfs.gene_coord.bed
awk 'BEGIN{FS=",";OFS="\t"}NR>1{print $6,$4-1,$5,$1,0,"+"}' \
    $data_dir/genes.txt > $output_dir/orfs.transcript_coord.bed
bedtools getfasta -s -fi $output_dir/transcripts.fa \
    -bed $output_dir/orfs.transcript_coord.bed \
    | awk '{gsub(/::.*/,""); print}' > $output_dir/orfs.fa
```

### Get SHAPE reactivities
```bash
# rename filenames to transcript IDs
for condition in cellfree incell kasugamycin;do
    [ -d "$output_dir/${condition}_SHAPE" ] || mkdir -p "$output_dir/${condition}_SHAPE"
    while IFS="," read tx_id strand start stop genes;do
        filename="$data_dir/${condition}_SHAPE/${start}-${stop}.shape"
        destname="$output_dir/${condition}_SHAPE/${tx_id}.shape"
        if [ -f "$filename" ];then
            echo "${filename} => $destname"
            cp $filename $destname
            if [ "$(wc -l < $filename)" -ne "$(($stop - $start + 1))" ];then
                echo "warning: number of nucleotides mismatch in $filename"
            fi
        fi
    done < $data_dir/transcripts.txt
done
# convert from shape format to HDF5 format
for condition in cellfree incell kasugamycin;do
    bin/shapemap.py shape_to_hdf5 -i $output_dir/${condition}_SHAPE -o $output_dir/${condition}_SHAPE.reactivities
done
for condition in cellfree incell kasugamycin;do
    bin/shapemap.py get_orf_reactivities -i $output_dir/${condition}_SHAPE.reactivities \
        --orf-file $output_dir/orfs.transcript_coord.bed \
        -o $output_dir/${condition}_SHAPE.orf.reactivities
done
```

## Structure inference using patteRNA

### Install patteRNA (requires Python 3)
```bash
git clone https://github.com/AviranLab/patteRNA
cd dist/
tar zxf patteRNA-latest.tar.gz
cd patteRNA-latest
python setup.py install
```

## Prepare SHAPE-MaP data
SHAPE-MaP format (.map):
Four columns: Nucleotide Index, SHAPE reactivity, SHAPE std err, Sequence

### Model RNAs
CT/dot files of known RNAs:

Directory of CT files used in RME:
`/Share/home/shibinbin/data/RME_structures/structure`.

SHAPE-MaP => CT file name used in RME
```
16S 16SRRNA-domain1,16SRRNA-domain2,16SRRNA-domain3,16SRRNA-domain4
5S 5SRRNA
23S 23SRRNA-domain1,23SRRNA-domain2,23SRRNA-domain3,23SRRNA-domain4,23SRRNA-domain5,23SRRNA-domain6
HCV_IRES HCV_domain2
TPP TPPribo-ecoli
```

```bash
model_RNAs='16SRRNA-domain1 16SRRNA-domain2 16SRRNA-domain3 16SRRNA-domain4 5SRRNA
    23SRRNA-domain1 23SRRNA-domain2 23SRRNA-domain3 23SRRNA-domain4 23SRRNA-domain5 23SRRNA-domain6
    HCV_domain2 TPPribo-ecoli'
{
for rna in $model_RNAs;do
    cat /Share/home/shibinbin/data/RME_structures/sequence/${rna}.seq
done
} | sed 's/^>\s*/>/' > output/predict_reactivity/model_RNAs_domains/sequences.fa
{
for rna in $model_RNAs;do
    cat /Share/home/shibinbin/data/RME_structures/structure/${rna}.dot
done
} | sed 's/^>\s*/>/' > output/predict_reactivity/model_RNAs_domains/structures.dot
```

**Map positions in SHAPE-MaP to positions in RME structures**
```bash
# convert CT files to dot files
for ct in $(ls ct/);do
    ~/apps/src/RNAstructure/exe/ct2dot ct/$ct 1 dot/${ct/.ct/.dot}
done
# convert dot files to FASTA files
for dot in $(ls dot/);do
    awk 'NR<=2' dot/$dot > sequence/${dot/.dot/.fa}
done
```

**Run patteRNA on SHAPE-MaP reactivities**
```bash
data_dir=/Share/home/shibinbin/data/SHAPE-MaP/Nature_Methods_2014/shape-map_DATA/model_RNAs_DATA/
output_dir=output/patteRNA/SHAPE-MaP
[ -d "$output_dir/model_RNAs.raw" ] || mkdir -p $output_dir/model_RNAs.raw
for rna in 16S 23S 5S Group_I Group_II HCV_IRES;do
    cp $data_dir/${rna}.1m7.map $output_dir/model_RNAs.raw/${rna}.map
done
cp $data_dir/TPP_1M7.map $output_dir/model_RNAs.raw/TPP.map
[ -d "$output_dir/model_RNAs" ] || mkdir -p $output_dir/model_RNAs
bin/shapemap.py shape_to_patterna -i $output_dir/model_RNAs.raw -x .map --na-values -999 -o $output_dir/model_RNAs.shape
bin/shapemap.py shape_to_patterna -i $output_dir/model_RNAs.raw -x .map --na-values -999 --sequence -o $output_dir/model_RNAs.fa
bin/shapemap.py shape_to_hdf5 -i output/patteRNA/SHAPE-MaP/${dataset}.raw -x .map -o $output_dir/${dataset}.reactivities.h5
# split model RNAs into domains
dataset=model_RNAs_domains
bin/shapemap.py patterna_input_to_hdf5 -i $output_dir/${dataset}.shape -o $output_dir/${dataset}.reactivities.h5
# train the model
patteRNA $output_dir/${dataset}.shape $output_dir/${dataset}.output -vl
# get posterior probabilities
patteRNA $output_dir/${dataset}.shape $output_dir/${dataset}.output \
    --model $output_dir/${dataset}.output/trained_model.pickle -vl --viterbi --posteriors
# convert posteriors/viterbi to HDF5 format
bin/shapemap.py patterna_to_hdf5 -i $output_dir/${dataset}.output -o $output_dir/${dataset}
```

### High-throughput SHAPE-MaP (Cell 2018)
```bash
data_dir=output/SHAPE-MaP/Cell_2018
output_dir=output/patteRNA/SHAPE-MaP
[ -d "$output_dir" ] || mkdir -p "$output_dir"
# prepare patteRNA input files
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    bin/shapemap.py shape_to_patterna -i $data_dir/$dataset -x .shape -o $output_dir/Cell_2018_${dataset}.shape
    bin/shapemap.py shape_to_patterna -i $data_dir/$dataset -x .shape --sequence -o $output_dir/Cell_2018_${dataset}.fa
done
# train and infer posteriors using the patteRNA model
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    patteRNA $output_dir/Cell_2018_${dataset}.shape $output_dir/Cell_2018_${dataset}.output -vl
    patteRNA $output_dir/Cell_2018_${dataset}.shape $output_dir/Cell_2018_${dataset}.output \
        --model $output_dir/Cell_2018_${dataset}.output/trained_model.pickle -vl --viterbi --posteriors
done
# convert posteriors/viterbi to HDF5 format
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    bin/shapemap.py patterna_to_hdf5 -i $output_dir/Cell_2018_${dataset}.output -o $output_dir/Cell_2018_${dataset}
done
# convert SHAPE-MaP raw reactivities to HDF5 format
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    bin/shapemap.py shape_to_hdf5 -i output/SHAPE-MaP/Cell_2018/${dataset} -x .shape -o $output_dir/Cell_2018_${dataset}.reactivities.h5
done
```

### Predict SHAPE-MaP
```bash
# binarize reactivities using percentiles
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    for percentile in 5 10 20 30;do
        bin/predict_reactivity.py binarize -i $output_dir/Cell_2018_${dataset}.reactivities.h5:reactivities \
            --method "percentile:${percentile},$((100 - $percentile))" \
            -o $output_dir/Cell_2018_${dataset}.percentile_${percentile}.h5
    done
done
# cross-validation split
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    bin/predict_reactivity.py cv_split -k 5 -i $output_dir/Cell_2018_${dataset}.reactivities.h5:reactivities \
        -o $output_dir/Cell_2018_${dataset}.cv_split.h5
done
# create input dataset (with binarization)
binarize_methods="percentile_5 percentile_10 percentile_20 percentile_30 viterbi"
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    rm -rf "$output_dir/Cell_2018_${dataset}.dataset"
    [ -d "$output_dir/Cell_2018_${dataset}.dataset" ] || mkdir -p "$output_dir/Cell_2018_${dataset}.dataset"
    for i in $(seq 0 4);do
        for binarize_method in $binarize_methods;do
            bin/predict_reactivity.py create_dataset -i $output_dir/Cell_2018_${dataset}.${binarize_method}.h5 \
                --sequence-file $output_dir/Cell_2018_${dataset}.fa \
                --cv-split-file $output_dir/Cell_2018_${dataset}.cv_split.h5:${i} \
                --window-size 128 --balanced \
                -o $output_dir/Cell_2018_${dataset}.dataset/w=128,b=${binarize_method},i=${i}
        done
    done
done
# create input dataset (raw reactivities)
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    bin/predict_reactivity.py create_dataset -i $output_dir/Cell_2018_${dataset}.reactivities.h5:reactivities \
        --sequence-file $output_dir/Cell_2018_${dataset}.fa \
        --window-size 128 \
        -o $output_dir/Cell_2018_${dataset}.dataset/w=128,b=reactivities
done
# train the model and evaluate
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    for model in basic mlp1 conv1 resnet1;do
        [ -d $output_dir/Cell_2018_${dataset}.keras_log ] || mkdir -p $output_dir/Cell_2018_${dataset}.keras_log
        [ -d $output_dir/Cell_2018_${dataset}.metrics ] || mkdir -p $output_dir/Cell_2018_${dataset}.metrics
        bin/predict_reactivity.py train -i $output_dir/Cell_2018_${dataset}.prediction/w=128 \
            --model-file $output_dir/Cell_2018_${dataset}.trained_models/w=128.m=${model} -m $model \
            --keras-log $output_dir/Cell_2018_${dataset}.keras_log/w=128.m=${model}
        bin/predict_reactivity.py evaluate -i $output_dir/Cell_2018_${dataset}.prediction/w=128 \
            --model-file $output_dir/Cell_2018_${dataset}.trained_models/w=128.m=${model} \
            -o $output_dir/Cell_2018_${dataset}.metrics/w=128.m=${model}
    done
done
# evaluate on model RNAs
[ -d "$output_dir/model_RNAs.predict" ] || mkdir -p "$output_dir/model_RNAs.predict"
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    for model in basic mlp1 conv1 resnet1;do
        bin/predict_reactivity.py predict -i $output_dir/model_RNAs.fa \
            --model-file $output_dir/Cell_2018_${dataset}.trained_models/w=128.m=${model} \
            -o $output_dir/model_RNAs.predict/d=Cell_2018_${dataset}.w=128.m=${model}
    done
done

[ -d "$output_dir/model_RNAs_domains.predict" ] || mkdir -p "$output_dir/model_RNAs_domains.predict"
for dataset in cellfree_SHAPE incell_SHAPE kasugamycin_SHAPE;do
    for model in basic mlp1 conv1 resnet1;do
        bin/predict_reactivity.py predict -i $output_dir/model_RNAs_domains.fa \
            --model-file $output_dir/Cell_2018_${dataset}.trained_models/w=128.m=${model} \
            -o $output_dir/model_RNAs_domains.predict/d=Cell_2018_${dataset}.w=128.m=${model}
    done
done

```

## Run the Snakemake pipeline
```bash
snakemake --snakefile workflows/predict_reactivity/Snakefile --configfile workflows/predict_reactivity/config.json
for f in $(find output/predict_reactivity/model_RNAs_domains/metrics_by_rna/ -type f);do
    mv $f ${f/./,}
done 
```