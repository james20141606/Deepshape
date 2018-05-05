## Summary of the icSHAPE datasets
**Lu_2016**

| SRA ID     | Description |
|:----------:|:-----------:|
| SRR3194440 | icSHAPE - 293T - DMSO rep1 |
| SRR3194441 | icSHAPE - 293T - DMSO rep2 |
| SRR3194442 | icSHAPE - 293T - NAI-N3 In Vitro rep1 |
| SRR3194443 | icSHAPE - 293T - NAI-N3 In Vitro rep2 |
| SRR3194444 | icSHAPE - 293T - NAI-N3 In Vivo rep1 |
| SRR3194445 | icSHAPE - 293T - NAI-N3 In Vivo rep2 |

**Spitale_2015**
| SRA ID     | Description |
|:----------:|:-----------:|
| SRR1534952 | v65 polyA(+) icSHAPE DMSO Biological Replicate 1 |
| SRR1534953 | v65 polyA(+) icSHAPE DMSO Biological Replicate 2 |
| SRR1534954 | v65 polyA(+) icSHAPE in vitro NAI-N3 Biological Replicate 1 |
| SRR1534955 | v65 polyA(+) icSHAPE in vitro NAI-N3 Biological Replicate 2 |
| SRR1534956 | v65 polyA(+) icSHAPE in vivo NAI-N3 Biological Replicate 1 |
| SRR1534957 | v65 polyA(+) icSHAPE in vivo NAI-N3 Biological Replicate 2 |

## Prepare data for BUMHMM
```bash
for data_name in Lu_2016_invitro Lu_2016_invivo Spitale_2015_invitro Spitale_2015_invivo;do
    if [ "$data_name" = "Lu_2016_invitro" ];then
        sequence_file=$HOME/data/genomes/fasta/Human/hg19.transcript.v19.fa
        bin/run_BUMHMM.py PrepareBumhmmForIcshape \
            --control-file output/icSHAPE_preprocess/$data_name/SRR3194440.rt \
            --control-file output/icSHAPE_preprocess/$data_name/SRR3194441.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR3194442.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR3194443.rt \
            --sequence-file $sequence_file \
            -o output/BUMHMM/icSHAPE/$data_name/input.h5
    elif [ "$data_name" = "Lu_2016_invivo" ];then
        sequence_file=$HOME/data/genomes/fasta/Human/hg19.transcript.v19.fa
        bin/run_BUMHMM.py PrepareBumhmmForIcshape \
            --control-file output/icSHAPE_preprocess/$data_name/SRR3194440.rt \
            --control-file output/icSHAPE_preprocess/$data_name/SRR3194441.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR3194444.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR3194445.rt \
            --sequence-file $sequence_file \
            -o output/BUMHMM/icSHAPE/$data_name/input.h5
    elif [ "$data_name" = "Spitale_2015_invitro" ];then
        sequence_file=$HOME/data/genomes/fasta/Mouse/mm10.transcript.vM12.fa
        bin/run_BUMHMM.py PrepareBumhmmForIcshape \
            --control-file output/icSHAPE_preprocess/$data_name/SRR1534952.rt \
            --control-file output/icSHAPE_preprocess/$data_name/SRR1534953.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR1534954.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR1534955.rt \
            --sequence-file $sequence_file \
            -o output/BUMHMM/icSHAPE/$data_name/input.h5
    elif [ "$data_name" = "Spitale_2015_invivo" ];then
        sequence_file=$HOME/data/genomes/fasta/Mouse/mm10.transcript.vM12.fa
        bin/run_BUMHMM.py PrepareBumhmmForIcshape \
            --control-file output/icSHAPE_preprocess/$data_name/SRR1534952.rt \
            --control-file output/icSHAPE_preprocess/$data_name/SRR1534953.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR1534956.rt \
            --treatment-file output/icSHAPE_preprocess/$data_name/SRR1534957.rt \
            --sequence-file $sequence_file \
            -o output/BUMHMM/icSHAPE/$data_name/input.h5
    fi

    Rscript bin/run_BUMHMM.R output/BUMHMM/icSHAPE/$data_name/input.h5 \
        output/BUMHMM/icSHAPE/$data_name/posteriors.values.h5 50
    bin/run_BUMHMM.py BumhmmToGenomicData \
        -i output/BUMHMM/icSHAPE/$data_name/posteriors.values.h5 \
        --bumhmm-input-file output/BUMHMM/icSHAPE/$data_name/input.h5 \
        -o output/BUMHMM/icSHAPE/$data_name/posteriors.h5
    bin/preprocess.py BaseDistribution \
        -i output/BUMHMM/icSHAPE/$data_name/posteriors.h5 \
        --feature bumhmm \
        --sequence-file $sequence_file \
        --cutoff1 0.4 --cutoff2 0.6 \
        -o output/BUMHMM/icSHAPE/$data_name/BaseDistribution.pdf
    bin/report.py CorrelationBetweenBumhmmAndIcshape \
        --bumhmm-file output/BUMHMM/icSHAPE/$data_name/posteriors.h5 \
        --icshape-file data/icSHAPE/$data_name/all.h5 \
        --prefix output/BUMHMM/icSHAPE/$data_name/CorrelationWithIcshape
    bin/report.py CompareBumhmmWithCoverageAndDropoff \
        -i output/BUMHMM/icSHAPE/$data_name/posteriors.values.h5 \
        --bumhmm-input-file output/BUMHMM/icSHAPE/$data_name/input.h5 \
        -o output/BUMHMM/icSHAPE/$data_name/CompareBumhmmWithCoverageAndDropoff.pdf
    bin/report.py BumhmmExamples \
        -i output/BUMHMM/icSHAPE/$data_name/posteriors.values.h5 \
        --bumhmm-input-file output/BUMHMM/icSHAPE/$data_name/input.h5 \
        -o output/BUMHMM/icSHAPE/$data_name/BumhmmExamples.pdf
    bin/report.py GenomicDataDistribution \
        --infile output/BUMHMM/icSHAPE/$data_name/posteriors.h5 \
        --feature bumhmm \
        --weight 1e-6 \
        --ylabel 'Counts (x1000000)' \
        --xlabel 'BUMHMM posterior probabilities' \
        --outfile output/BUMHMM/icSHAPE/$data_name/BumhmmDistribution.pdf
done
```
## Modulate the icSHAPE RT stop counts to audio signal
```bash
bin/run_BUMHMM.py IcshapeRtToWav \
    -i output/icSHAPE_preprocess/Lu_2016_invitro/SRR3194440.rt \
    -o tmp/Lu_2016_invitro.wavfiles
```

## Separate the datasets by region
```bash
gencode_version=v19
data_names="Lu_2016_invitro Lu_2016_invivo Spitale_2015_invitro Spitale_2015_invivo"
data_names="Spitale_2015_invitro Spitale_2015_invivo"
for data_name in $data_names;do
    if [ "$data_name " = "Lu_2016_invitro" ] || [ "$data_name" = "Lu_2016_invivo" ];then
        gencode_version=v19
    elif [ "$data_name" = "Spitale_2015_invitro" ] || [ "$data_name"  = "Spitale_2015_invivo" ];then
        gencode_version=vM12
    fi
    ln -s -f ../../../output/BUMHMM/icSHAPE/$data_name/posteriors.h5 \
        data/icSHAPE/${data_name}.BUMHMM/all.h5
    # create subsets of the data by genomic regions
    [ -d "data/icSHAPE/${data_name}.BUMHMM" ] || mkdir -p "data/icSHAPE/${data_name}.BUMHMM"
    for region in CDS 5UTR 3UTR lncRNA miRNA ncRNA;do
        bin/preprocess.py SubsetGenomicDataByRegion \
            -i data/icSHAPE/${data_name}.BUMHMM/all.h5 \
            --region-file $HOME/data/gtf/gencode.${gencode_version}/${region}.transcript.bed \
            --feature bumhmm \
            -o data/icSHAPE/${data_name}.BUMHMM/${region}.h5
    done
done
```
