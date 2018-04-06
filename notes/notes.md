## preprocess

1. Create datasets to for further 2D data  X: n*128*4 y:n*128  train and test
{
region=all
coverage=0.9
echo HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/Spitale_2015_invivo/${region} \
--sequence-file ~/shibinbin/data/gtf/gencode.vM12/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 128 \
--stride 16 \
--train-test-split 1 \
--percentile 5 \
--dense-output \
-o new/Spitale_2015_invivo_${region}_${coverage}
} > Jobs/data.txt
qsubgen -n data -q Z-LU -a 1-1 -j 1 --bsub --task-file Jobs/data.txt
bsub < Jobs/data.sh

{
region=all
file=Spitale_2015_invivo
seqfile=vm12
coverage=0.1
bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/${file}/${region} \
--sequence-file data/gtf/${seqfile}/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 16 \
--stride 16 \
--train-test-split 1 \
--percentile 5 \
--dense-output \
-o data/new/${file}/${file}_${region}_16_stride16
}
all  3UTR  5UTR   CDS lncRNA  miRNA  ncRNA Spitale_2015_invitro   Spitale_2015_invivo   vm12 
Lu_2016_invitro_hg38  Lu_2016_invivo_hg38  v26
 
{
for region in 'all' '3UTR' '5UTR' 'CDS' 'lncRNA'  'miRNA'  'ncRNA'; do
for file in 'Lu_2016_invitro_hg38'  'Lu_2016_invivo_hg38'; do
seqfile=v26
coverage=0.1
HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/${file}/${region} \
--sequence-file data/gtf/${seqfile}/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 128 \
--stride 1 \
--train-test-split 0.8 \
--percentile 5 \
--dense-output \
-o data/new/${file}/${file}_${region}_${coverage}
done
done
}

## use coverage 0.5 and have all data pipeline   4.4
{
for region in 'all'; do
for file in 'Lu_2016_invitro_hg38'  'Lu_2016_invivo_hg38'; do
seqfile=v26
coverage=0.5
HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/${file}/${region} \
--sequence-file data/gtf/${seqfile}/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 128 \
--stride 1 \
--train-test-split 0.999 \
--percentile 10 \
--dense-output \
-o data/new/alldata/${file}/${file}_${region}_${coverage}
done
done
}
{
for region in 'all'; do
for file in 'Spitale_2015_invitro' 'Spitale_2015_invivo'; do
seqfile=vm12
coverage=0.5
HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/${file}/${region} \
--sequence-file data/gtf/${seqfile}/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 128 \
--stride 1 \
--train-test-split 0.999 \
--percentile 10 \
--dense-output \
-o data/new/alldata/${file}/${file}_${region}_${coverage}
done
done
}
#4.4 coverage0.5下所有的转成图片，这次直接加上了y_train，改了脚本
  
{
for dir in 'Lu_2016_invitro_hg38' 'Lu_2016_invivo_hg38' 'Spitale_2015_invitro' 'Spitale_2015_invivo'; do
for region in 'all'; do
bin/extract_2d_train.py \
  -i data/new/alldata/${dir}/${dir}_${region}_0.5 \
	-o data/new/alldata/train/train_${dir}_${region} \
	-s 128
done
done
} | parallel -P 6


2 create 2D data


{
region=CDS
coverages=$(seq 0.1 0.1 0.8)
pencentiles=$(seq 5 5 30)
for coverage in $coverages;do
for percentile in $pencentiles;do
echo HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/Spitale_2015_invivo/${region} \
--sequence-file ~/shibinbin/data/gtf/gencode.vM12/sequences/${region}.transcript.fa \
--min-coverage ${coverage} \
--window-size 128 \
--stride 1 \
--train-test-split 0.8 \
--percentile ${percentile} \
--dense-output \
-o new/Spitale_2015_invivo_${region}_${coverage}_${percentile}
done
done
} > Jobs/data_1.txt
qsubgen -n data_1 -q Z-LU -a 1-8 -j 7 --bsub --task-file Jobs/data_1.txt
bsub < Jobs/data_1.sh

#转图片

{
for dir in 'Lu_2016_invitro_hg38'  'Lu_2016_invivo_hg38' 'Spitale_2015_invitro' 'Spitale_2015_invivo'; do
 mkdir ${dir}/train 
done
}
{
for dir in 'Lu_2016_invitro_hg38'  'Lu_2016_invivo_hg38' 'Spitale_2015_invitro' 'Spitale_2015_invivo'; do
 rm data/new/${dir}/train/* 
done
}

{
for dir in 'Spitale_2015_invivo'; do
for region in 'all' '3UTR' '5UTR' 'CDS' 'lncRNA'; do
bin/extract_2d_train.py \
  -i data/new/${dir}/${dir}_${region}_0.1 \
	-o data/new/${dir}/train/train_${region}
done
done
} | parallel -P 9
'Lu_2016_invitro_hg38'  'Lu_2016_invivo_hg38' 'Spitale_2015_invitro' 

{
bin/extract_2d_train.py \
  -i data/new/Spitale_2015_invivo/Spitale_2015_invivo_CDS_16_stride16  \
	-o data/new/Spitale_2015_invivo/train/train_CDS_winsize16_stride16 \
	-s 16
} | parallel -P 3

{
bin/extract_2d_train.py \
  -i data/new/Spitale_2015_invivo/Spitale_2015_invivo_all_128_stride16  \
	-o data/new/Spitale_2015_invivo/train/train_all_winsize128_stride16 \
	-s 128
} | parallel -P 3

{
bin/extract_2d_train.py \
  -i data/new/Spitale_2015_invivo/Spitale_2015_invivo_all_16_stride16  \
	-o data/new/Spitale_2015_invivo/train/train_all_winsize16_stride16 \
	-s 16
} | parallel -P 3

## 测试nan影响
(1)
#挑选percentile 25，不加coverage   然后在每条上面随机加10% nan  以此类推
{
region=CDS
echo HDF5_USE_FILE_LOCKING=FALSE bin/preprocess.py CreateDatasetFromGenomicData \
-i data/icSHAPE/Spitale_2015_invivo/${region} \
--sequence-file ~/shibinbin/data/gtf/gencode.vM12/sequences/${region}.transcript.fa \
--window-size 128 \
--stride 1 \
--train-test-split 0.8 \
--percentile 25 \
--dense-output \
-o new/Spitale_2015_invivo_${region}_percentile_25
} > Jobs/data_nan.txt
qsubgen -n data_nan -q Z-LU -a 1-1 -j 1 --bsub --task-file Jobs/data_nan.txt
bsub < Jobs/data_nan.sh

#随机nan  每次加10%    0.5  0.6 0.7 0.8 0.9

new/Spitale_2015_invivo_CDS_percentile_25
new/y_Spitale_2015_invivo_CDS_percentile_25_${coverage}

只输出y！！


#图片不用动的




#run unet 分别测试  循环着写
coverage要手动改，不并行
python ./run_unet_128.py \
    -a 10 \
    -e 1 \
    -n 10000 \
    -i data/new/train_0
    -j data/new/test_0
    -y data/new/y_Spitale_2015_invivo_CDS_percentile_25_0.5
    
# nucleotide mutation    
bash
bin/preprocess.py GenerateMutatedSequences \
    -i data/gtf/gencode.vM12/sequences/CDS.transcript.fa \
    -o mutation/Spitale_2015_invivo_CDS.mutate_and_map.fa


#mutation map

python ./mutate_and_map.py \
	-i 0 \
	-c 10


#2d structural data
2d_structure_data.ipynb
异常处理


awk '{print $2}' telomerase_AC121792.109554-109950.ct
awk '{print $5}' telomerase_AC121792.109554-109950.ct
{
for name in  'telomerase_AC121792.109554-109950.ct' 'telomerase_AF147806.247-692.ct' 'telomerase_AF221908.136-584.ct' 'telomerase_AF221911.104-513.ct' 'telomerase_AF221913.109-520.ct' 'telomerase_AF221914.112-589.ct' 'telomerase_AF221919.107-614.ct' 'telomerase_AF221920.99-541.ct' 'telomerase_AF221927.113-556.ct' 'telomerase_AF221933.121-679.ct' 'telomerase_AF221940.103-499.ct' ; do
awk '{print $2}' ${name} > ../exception/${name}_seq.txt
awk '{print $5}' ${name} > ../exception/${name}_match.txt
done
}


#final acc 4.4
{
python bin/final_acc.py \
	-m output/unet_allstride16_4.3_norestrict.hdf5 \
	-s output/acc/acc_unet_allstride16_4.3_norestrict \
	-c 540 \
  -p 1 \
  -g 4 \
  -i known/pictures \
}
{
python bin/final_acc.py \
	-m output/unet_allstride16_4.3_restrict.hdf5 \
	-s output/acc/acc_unet_unet_allstride16_4.3_restrict.hdf5 \
	-c 540 \
  -p 1 \
  -i known/pictures_540 \
  -g 4
}
4.5
{
python bin/final_acc.py \
	-m output/unet_allstride1_4.4_restrict.hdf5 \
	-s output/acc/acc_unet_allstride1_4.4_restrict \
	-c 540 \
  -p 1 \
  -g 4 \
  -i known/pictures_540 
}

#4.6
run 2d unet model
{
python bin/run_2d_structure_unet.py \
	-m output/unet_2d_4.6.hdf5 \
	-s output/unet_2d_4.6.hdf5 \
	-c 400 \
  -e 100 \
  -g 4 \
  -i known/2d/training.hdf5 
}

run rnnseq2seq and attention
{
python bin/run_rnnseq2seq.py \
  -s output/rnnseq2seq_model4.hdf5 \
  -e 10 \
  -g 3 \
  -i data/new/alldata/seqdata/all1.hdf5 \
  -model-ind 4 \
  -en-depth 4 \
  -de-depth 5 \
  -dep 4 \
  -hid-dim 10 
}

{
python bin/run_rnnseq2seq.py \
  -s output/rnnseq2seq_model2.hdf5 \
  -e 10 \
  -g 4 \
  -i data/new/alldata/seqdata/all2.hdf5 \
  -model-ind 2 \
  -en-depth 4 \
  -de-depth 5 \
  -dep 4 \
  -hid-dim 10 
}

{
python bin/run_rnnseq2seq.py \
  -s output/rnnseq2seq_model3.hdf5 \
  -e 10 \
  -g 5 \
  -i data/new/alldata/seqdata/all3.hdf5 \
  -model-ind 3 \
  -en-depth 4 \
  -de-depth 5 \
  -dep 4 \
  -hid-dim 10 
}

unet_allstride16_4.3_norestrict.hdf5
unet_allstride16_4.3_restrict.hdf5


