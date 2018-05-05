## Nucleotide periodicity in CDS regions
```bash
bin/analyze_data.py nucleotide_periodicity \
    -i data/gtf/gencode.v19/sequences/CDS.transcript.fa \
    -o output/analyze_data/nucleotide_periodicity/gencode.v19.CDS.pdf
bin/analyze_data.py nucleotide_periodicity \
    -i data/gtf/gencode.vM12/sequences/CDS.transcript.fa \
    -o output/analyze_data/nucleotide_periodicity/gencode.vM12.CDS.pdf
bin/analyze_data.py nucleotide_periodicity \
    -i output/SHAPE-MaP/Cell_2018/orfs.fa \
    -o output/analyze_data/nucleotide_periodicity/e_coli.CDS.pdf
```

## Reactivity periodicity in CDS regions
```bash
for condition in cellfree incell kasugamycin;do
    bin/analyze_data.py reactivity_periodicity \
        --assay-type shapemap \
        -i output/SHAPE-MaP/Cell_2018/${condition}_SHAPE.orf.reactivities \
        -o output/analyze_data/reactivity_periodicity/e_coli.${condition}.CDS.pdf
done
for dataset in Lu_2016_invitro Lu_2016_invivo Spitale_2015_invitro Spitale_2015_invivo;do
    bin/analyze_data.py reactivity_periodicity \
        --assay-type icshape \
        -i data/icSHAPE/$dataset/CDS \
        -o output/analyze_data/reactivity_periodicity/${dataset}.CDS.pdf
done
```

## Background periodicity in CDS regions
```bash
datasets="Lu_2016_invitro Lu_2016_invivo Spitale_2015_invitro Spitale_2015_invivo"
for dataset in $datasets;do
    if [ "$dataset" = "Lu_2016_invitro" ] || [ "$dataset" = "Lu_2016_invivo" ];then
        annotation=gencode.v19
    elif [ "$dataset" = "Spitale_2015_invitro" ] || [ "$dataset" = "Spitale_2015_invivo" ];then
        annotation=gencode.vM12
    fi
    # normalize RT-stop counts by base density
    bin/preprocess.py NormalizeRtStopByBaseDensity \
        -i data/icSHAPE/$dataset/raw/background_normalized \
        -o output/datasets/icSHAPE/$dataset/background_rt_stop.all
    # create subsets of the data by genomic regions
    bin/preprocess.py SubsetGenomicDataByRegion \
        -i output/datasets/icSHAPE/$dataset/background_rt_stop.all \
        --region-file data/gtf/$annotation/CDS.transcript.bed \
        --feature rt_stop \
        -o output/datasets/icSHAPE/$dataset/background_rt_stop.CDS
done
# plot periodicity of background RT-stop
for dataset in $datasets;do
    bin/analyze_data.py reactivity_periodicity \
        --assay-type rt_stop \
        -i output/datasets/icSHAPE/$dataset/background_rt_stop.CDS \
        -o output/analyze_data/reactivity_periodicity/${dataset}.background_rt_stop.CDS.pdf
done
```

## 5'-end periodicity in RNA-seq data
```bash
bam_file=/Share/home/xuzhiyu/ribo-seq/new_update/brain_tumor/bam/normal.mRNA.sort.bam
bedtools bamtobed -i $bam_file | tee >(awk -F'\t' '$5==255{print $1,$2,$2+1}' )
genomic_signal_tools.py bam_to_hdf5 -i $bam_file --offset 0 --chrom-sizes-file data/chrom_sizes/hg19 -o tmp/rna_seq.5p.h5
```