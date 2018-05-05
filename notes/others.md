## Remove ZW:f:xx from transcript BAM files
```bash
samtools view -h output/RNA-seq/mESC.mm9/SRR5171019.transcript.bam | sed 's/\tZW:f:[0-9\.]\+//' \
    | head -n 1000000 | samtools view -h -b -o tmp/transcript.bam
rsem-tbam2gbam data/rsem_ref/mm9.gencode_vM1/rsem tmp/transcript.bam tmp/genome.bam
run data/rsem_ref/mm9.gencode_vM1/rsem tmp/transcript.bam tmp/genome.bam
~/apps/src/RSEM-1.3.0-patched/rsem-tbam2gbam data/rsem_ref/mm9.gencode_vM1/rsem tmp/transcript.bam tmp/genome.bam
~/pkgs/RSEM/1.3.0/bin/rsem-tbam2gbam data/rsem_ref/mm9.gencode_vM1/rsem tmp/transcript.bam tmp/genome.bam
```