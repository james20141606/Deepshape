## Find rRNA genes in the annotation file
```bash
[ -d "output/infernal" ] || mkdir -p output/infernal
awk 'BEGIN{FS="\t";OFS="\t"} {
    if($3 != "transcript") {next}
    match($9, /transcript_id "([^"]+)"/, tid);
    match($9, /gene_name "([^"]+)"/, gn);
    match($9, /gene_type "([^"]+)"/, gt);
    if(gt[1] == "rRNA") print tid[1],gn[1]
}' $HOME/data/gtf/gencode.v19/gencode.v19.annotation.gtf > output/infernal/rRNA_list.txt
```
## Extract the CM model of 5SrRNA from `Rfam.cm`
```bash
export PATH=$PATH:$HOME/pkgs/infernal/1.1.2/binaries/
[ -d cm ] | mkdir cm
cmfetch Rfam.cm 5S_rRNA > cm/5S_rRNA.cm
```
## Align the sequences to the 5S rRNA CM
```bash
awk '{if($2 ~/^RNA5S|^5S_rRNA/) print $1}' output/infernal/rRNA_list.txt \
    | $HOME/projects/tests/python/fasta_fetch.py -f - $HOME/data/genomes/fasta/Human/hg19.transcript.v19.fa \
    > output/infernal/5S_rRNA.fa
awk '{if($2 ~/^RNA5S|^5S_rRNA/) print $1}' output/infernal/rRNA_list.txt > output/infernal/5S_rRNA.transcript_id.txt
[ -d "output/infernal/cmalign" ] || mkdir -p output/infernal/cmalign
for transcript_id in $(cat output/infernal/5S_rRNA.transcript_id.txt);do
    $HOME/projects/tests/python/fasta_fetch.py --ids $transcript_id output/infernal/5S_rRNA.fa \
    | cmalign -o output/infernal/cmalign/${transcript_id}.sto $HOME/data/Rfam/12.2/cm/5S_rRNA.cm -
done
```
