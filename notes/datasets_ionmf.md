## Download CLIP data
CLIP data are from iONMF

**Reduced dataset:**
```bash
git clone https://github.com/mstrazar/iONMF.git
```

**Full dataset**
```bash
git clone -b master-full https://github.com/mstrazar/iONMF.git
```

### HEK293 datasets (all mapped to hg19)

```
10_PARCLIP_ELAVL1A_hg19
11_CLIPSEQ_ELAVL1_hg19
12_PARCLIP_EWSR1_hg19
13_PARCLIP_FUS_hg19
14_PARCLIP_FUS_mut_hg19
15_PARCLIP_IGF2BP123_hg19
1_PARCLIP_AGO1234_hg19
21_PARCLIP_MOV10_Sievers_hg19
22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome
23_PARCLIP_PUM2_hg19
24_PARCLIP_QKI_hg19
25_CLIPSEQ_SFRS1_hg19
26_PARCLIP_TAF15_hg19
2_PARCLIP_AGO2MNASE_hg19
3_HITSCLIP_Ago2_binding_clusters
4_HITSCLIP_Ago2_binding_clusters_2
5_CLIPSEQ_AGO2_hg19
8_PARCLIP_ELAVL1_hg19
9_PARCLIP_ELAVL1MNASE_hg19
```

**Run the workflow**

```bash
snakemake --snakefile workflows/motif_ionmf/Snakefile \
    --configfile workflows/motif_ionmf/config.json
```
