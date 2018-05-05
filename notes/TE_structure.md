## Get genomic locations of TEs
### Download from UCSC Table Browser
URL: (https://genome.ucsc.edu/cgi-bin/hgTables)
Database: hg38, group: Repeats, track: RepeatMasker, table: rmsk
Database: mm10, group: Variation and Repeats, track: RepeatMasker, table: rmsk

### Repeatmasker table schema
```
field	example	SQL type	description
bin	607	smallint(5) unsigned	Indexing field to speed chromosome range queries.
swScore	12955	int(10) unsigned	Smith Waterman alignment score
milliDiv	105	int(10) unsigned	Base mismatches in parts per thousand
milliDel	9	int(10) unsigned	Bases deleted in parts per thousand
milliIns	10	int(10) unsigned	Bases inserted in parts per thousand
genoName	chr1	varchar(255)	Genomic sequence name
genoStart	3000000	int(10) unsigned	Start in genomic sequence
genoEnd	3002128	int(10) unsigned	End in genomic sequence
genoLeft	-192469843	int(11)	-#bases after match in genomic sequence
strand	-	char(1)	Relative orientation + or -
repName	L1_Mus3	varchar(255)	Name of repeat
repClass	LINE	varchar(255)	Class of repeat
repFamily	L1	varchar(255)	Family of repeat
repStart	-3055	int(11)	Start (if strand is +) or -#bases after match (if strand is -) in repeat sequence
repEnd	3592	int(11)	End in repeat sequence
repLeft	1466	int(11)	-#bases after match (if strand is +) or start (if strand is -) in repeat sequence
id	1	char(1)	First digit of id field in RepeatMasker .out file. Best ignored.
```

### Get repeat families and classes
```bash
for genome in mm10 hg38;do
    cat data/UCSC/$genome/RepeatMasker.txt | awk 'BEGIN{OFS="\t"}NR>1{print $11,$12,$13}' \
        | sort | uniq > data/UCSC/$genome/RepeatMasker.classes.txt
done
```
output is a table with 3 columns: repName, repClass, repFamily

## Get TE sequences
```bash
annotation=gencode.vM12
genome=mm10
dataset=Spitale_2015_invitro

annotation=gencode.v26
genome=hg38
dataset=Lu_2016_invitro
output_dir=output/TE/${dataset}.${genome}
[ -d "$output_dir" ] || mkdir -p $output_dir
# get subset of repeat classes/families
awk '($2 != "Unknown") && ($2 != "Simple_repeat") && ($0 !~ /\?/)' \
    data/UCSC/$genome/RepeatMasker.classes.txt \
    > $output_dir/RepeatMasker.classes.txt
# get genomic intervals of selected repeats
awk 'BEGIN{OFS="\t"} FNR==NR{names[$1]=1;next} {if($4 in names) print}' \
    $output_dir/RepeatMasker.classes.txt data/UCSC/$genome/RepeatMasker.bed  \
    > $output_dir/RepeatMasker.bed
# get genomic intervals of transcripts
awk 'BEGIN{FS="\t";OFS="\t"} NR==FNR{rna_ids[$1]=1;next} {if($4 in rna_ids) print $0}' \
    output/icSHAPE_preprocess/${dataset}/icshape.out \
    data/gtf/${annotation}/annotations/transcript.bed \
    | bedtools sort > $output_dir/transcripts.bed
# map TE types to transcripts
bedtools map -a $output_dir/transcripts.bed \
    -b $output_dir/RepeatMasker.bed \
    -c 4,4 -o count,collapse > $output_dir/transcript_TE.bed
# get genomic intervals of transcript exons
awk 'BEGIN{FS="\t";OFS="\t"} NR==FNR{rna_ids[$1]=1;next} {if($4 in rna_ids) print $0}' \
    output/icSHAPE_preprocess/${dataset}/icshape.out \
    data/gtf/${annotation}/annotations/exon.transcript.bed \
    | bedtools sort > $output_dir/exon.transcript.bed
# map transcript IDs to TEs
bedtools map -s -a $output_dir/RepeatMasker.bed \
    -b $output_dir/exon.transcript.bed \
    -f 1 -c 4 -o distinct | awk '$NF!="."' > $output_dir/TE_transcript.bed
# summarize TE types that can be mapped to transcripts
awk 'BEGIN{OFS="\t"}{n[$4]++}END{for(t in n){print t,n[t]}}' $output_dir/TE_transcript.bed \
    > $output_dir/TE_type_summary.txt
# get sequences of TEs mapped to transcripts
bedtools getfasta -name -s -fi data/genomes/${genome}/genome.fa \
    -bed $output_dir/TE_transcript.bed \
    > $output_dir/TE_transcript.fa
# get transcriptomic locations of TEs
# convert genePred format to BED12 format
awk 'BEGIN{OFS="\t"}{
    exonCount=$8
    split($9,exonStarts,",");
    blockStarts="";
    blockSizes="";
    for(i=1;i<=exonCount;i++) {if(i>1){blockStarts=blockStarts ","} blockStarts=blockStarts exonStarts[i]-$4}
    split($10,exonEnds,",");
    for(i=1;i<=exonCount;i++) {if(i>1){blockSizes=blockSizes ","} blockSizes=blockSizes exonEnds[i]-exonStarts[i]}
    print $2,$4,$5,$1,0,$3,0,0,0,exonCount,blockSizes,blockStarts
}' data/gtf/${annotation}/${annotation}.annotation.genePred \
    > $output_dir/transcripts.all.bed12
# get transcript sequences
bedtools getfasta -s -name -split -fi data/genomes/${genome}/genome.fa \
    -bed $output_dir/transcripts.all.bed12 \
    | sed 's/^\(>[^:]\+\).*/\1/' > $output_dir/transcripts.fa
samtools faidx $output_dir/transcripts.fa
# get genomic intervals of mapped transcripts BED12 file
awk 'BEGIN{FS="\t";OFS="\t"} NR==FNR{rna_ids[$1]=1;next} {if($4 in rna_ids) print $0}' \
    output/icSHAPE_preprocess/${dataset}/icshape.out \
    $output_dir/transcripts.all.bed12 \
    | bedtools sort > $output_dir/transcripts.bed12
# treat each TE as separate entries when overlapping with multiple transcripts
awk 'BEGIN{OFS="\t"}{n=split($7,ids,",");for(i=1;i<=n;i++) {$7=ids[i];print}}' \
    $output_dir/TE_transcript.bed > $output_dir/TE_transcript_separate.bed
awk 'BEGIN{OFS="\t"}FNR==NR{txStart[$4]=$2;blockCount[$4]=$10;blockSizes[$4]=$11;blockStarts[$4]=$12;next}
    {txid=$7;
    split(blockStarts[txid],blockStartsList,",");
    split(blockSizes[txid],blockSizesList,",");
    for(i=1;i<=blockCount[txid];i++) blockEndsList[i]=blockStartsList[i]+blockSizesList[i];
    txStarts[1]=0
    for(i=2;i<=blockCount[txid];i++) txStarts[i]=txStarts[i-1]+blockSizesList[i-1]
    txLength=txStarts[blockCount[txid]]+blockSizesList[blockCount[txid]]
    teStart=$2-txStart[txid];
    for(i=1;i<=blockCount[txid];i++){if(teStart<blockEndsList[i]){start=teStart-blockStartsList[i]+txStarts[i];break}}
    teEnd=$3-txStart[txid];
    for(i=1;i<=blockCount[txid];i++){if(teEnd<=blockEndsList[i]){end=teEnd-blockStartsList[i]+txStarts[i];break}}
    if($6=="-"){t=txLength-start;start=txLength-end;end=t}
    print txid,start,end,$4,$5,"+"
    }' $output_dir/transcripts.bed12 $output_dir/TE_transcript_separate.bed \
    > $output_dir/TE_transcript_separate.transcript_coord.bed
# get TE sequences from transcripts
bedtools getfasta -name -fi $output_dir/transcripts.fa \
    -bed $output_dir/TE_transcript_separate.transcript_coord.bed \
    > $output_dir/TE_transcript_separate.fa
bedtools getfasta -name -s -fi data/genomes/${genome}/genome.fa \
    -bed $output_dir/TE_transcript_separate.bed \
    > tmp/TE_transcript_separate.fa
# get icSHAPE scores of TE regions
bin/te_structure.py get_te_icshape\
    --bed-file $output_dir/TE_transcript_separate.transcript_coord.bed \
    --icshape-file data/icSHAPE/${dataset}/all \
    -o $output_dir/TE_transcript_separate.icshape
```

### Split TE fasta file by TE types
```bash
[ -d "$output_dir/TE_transcript_separate" ] || mkdir -p "$output_dir/TE_transcript_separate"
awk -v d=$output_dir/TE_transcript_separate '/^>/{name=substr($0, 1);header=$0;if(match($0,/^>([^:]+)/,a)>0){type=a[1]};next} 
{of=d "/" type;print header >> of;print $0 >> of}' $output_dir/TE_transcript_separate.fa
```
### Fetch RNA sequences that belong to a family
```bash
te_type=U1
awk -v type=$te_type '/^>/{name=substr($0, 1);header=$0;if(match($0,/^>([^:]+)/,a)>0){a_type=a[1]};next} 
{if(a_type==type){print header;print $0}}' $output_dir/TE_transcript_separate.fa
```

## Align sequences with Dfam
### Download Dfam HMM library
URL: (http://www.dfam.org/web_download/Release/Dfam_2.0/Dfam.hmm.gz)

### Download HMMER
URL: (http://hmmer.org/download.html)

### Download dfamscan
URL: (http://www.dfam.org/web_download/Tools/dfamscan.pl)

### Fetch separate HMM files
```bash
for te_type in `awk '{print $1}' output/TE/${dataset}.${genome}/RepeatMasker.classes.txt`;do
    echo "Fetch HMM model for $te_type"
    hmmfetch data/Dfam/2.0/Dfam.hmm $te_type > data/Dfam/2.0/hmm/${te_type}.hmm 2> /dev/null
    if [ "$?" -ne 0 ];then
        rm -f data/Dfam/2.0/hmm/${te_type}.hmm
    fi
done
```
### Align TE sequences to HMM models
```bash
[ -d "$output_dir/TE_transcript_separate.hmmalign" ] || mkdir -p $output_dir/TE_transcript_separate.hmmalign
{
for te_type in $(awk '{print $4}' $output_dir/TE_transcript_separate.bed);do
    if [ -f "data/Dfam/2.0/hmm/${te_type}.hmm" ];then
        echo hmmalign -o $output_dir/TE_transcript_separate.hmmalign/${te_type}.sto \
        data/Dfam/2.0/hmm/${te_type}.hmm $output_dir/TE_transcript_separate/$te_type
    fi
done
} | parallel -j20
```

### Summarize some statistics from the HMM alignment file for each TE type
```bash
{
for te_type in $(awk '{print $4}' $output_dir/TE_transcript_separate.bed);do
    if [ -f "$output_dir/TE_transcript_separate.hmmalign/${te_type}.sto" ];then
        echo bin/te_structure.py hmm_align_stats \
            -i $output_dir/TE_transcript_separate.hmmalign/${te_type}.sto \
            -o $output_dir/TE_transcript_separate.hmmalign/${te_type}.stats.txt
    fi
done
} | parallel -j20
```

### Plot HMM alignments for each TE type
```bash
{
for te_type in $(awk '{print $4}' $output_dir/TE_transcript_separate.bed);do
    if [ -f "$output_dir/TE_transcript_separate.hmmalign/${te_type}.sto" ];then
        echo bin/te_structure.py plot_alignment \
            -i $output_dir/TE_transcript_separate.hmmalign/${te_type}.sto \
            -o $output_dir/TE_transcript_separate.hmmalign/${te_type}.alignment_plot.pdf
    fi
done
} | parallel -j20
```

### Create table for HMM alignment statistics
```bash
{
header=1
for stats_file in $output_dir/TE_transcript_separate.hmmalign/*.stats.txt;do
    awk -v header=$header 'BEGIN{OFS="\t";n_keys=1}{keys[n_keys]=$1;n_keys++;vals[$1]=$2;}
    END{if(header==1) {for(i=1;i<=n_keys;i++) {if(i>1){printf "\t"} printf keys[i]} printf "\n"}
    for(i=1;i<=n_keys;i++){if(i>1){printf "\t"} printf vals[keys[i]]} printf "\n"}' $stats_file
    if [ "$header" -eq 1 ];then
        header=0
    fi
done
} > $output_dir/TE_transcript_separate.hmmalign.stats.txt
# plot stats
bin/te_structure.py plot_hmm_align_stats \
    -i $output_dir/TE_transcript_separate.hmmalign.stats.txt \
    --repclass-file data/UCSC/$genome/RepeatMasker.classes.txt \
    -o $output_dir/TE_transcript_separate.hmmalign.stats.pdf
```

### Map reactivities to HMM alignments
```bash
# remove empty alignment files
for f in output/TE/${dataset}.${genome}/TE_transcript_separate.hmmalign/*.sto;do
    if [ "$(wc -l < $f)" -eq 0 ];then
        rm -f $f
    fi
done

bin/te_structure.py hmm_align_reactivities \
    --feature icshape \
    --reactivity-file data/icSHAPE/$dataset/all \
    --alignment-dir $output_dir/TE_transcript_separate.hmmalign \
    -o $output_dir/TE_transcript_separate.hmmalign.reactivities
# plot aligned reactivities
bin/te_structure.py plot_hmm_align_reactivities \
    -i $output_dir/TE_transcript_separate.hmmalign.reactivities \
    -o $output_dir/TE_transcript_separate.hmmalign.reactivities.pdf
```

## Predict known structure families using Rfam
### Download latest Rfam
URL: (ftp://ftp.ebi.ac.uk/pub/databases/Rfam/13.0/)
Covariance models: (ftp://ftp.ebi.ac.uk/pub/databases/Rfam/13.0/Rfam.cm.gz)

### Fetch covariance models structure families
```bash
cmfetch Rfam.cm RF00003 > cm/U1.cm
```

### Align sequences to covariance models
```bash
te_type=U1
cmalign data/Rfam/13.0/cm/${te_type}.cm $output_dir/TE_transcript_separate/${te_type}
```

```bash
gffread gencode.v27.annotation.gtf -o  gencode.v27.annotation.gffread.gff3
igvtools sort gencode.v27.annotation.gffread.gff3 gencode.v27.annotation.gffread.sorted.gff3
igvtools index gencode.v27.annotation.gffread.sorted.gff3
```