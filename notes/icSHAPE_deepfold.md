## preprocess

1. Create datasets for Deepfold

Add a task (`CreateDatasetForIcshape`)in `create_jobs.py`.

Then run the job:
```bash
rm -rf data/icSHAPE/*/deepfold
rm -rf status/CreateDatasetForIcshape
bin/create_jobs.py CreateBsubJob -t CreateDatasetForIcshape
bin/create_jobs.py CheckStatus -t CreateDatasetForIcshape
rm -rf status/CreateDatasetForIcshapeDense
bin/create_jobs.py CreateBsubJob -t CreateDatasetForIcshapeDense
bin/create_jobs.py CheckStatus -t CreateDatasetForIcshapeDense
```

## Train the models
* Clean old files:
```bash
rm -rf keras_log/icSHAPE trained_models/icSHAPE metrics/icSHAPE
rm -rf status/TrainDeepfold1DForIcshape
rm -rf status/TrainDeepfold1DForIcshapeDense
rm -rf status/TrainDeepfold1DForIcshapeLogreg
rm -rf status/TrainDeepfold1DForIcshapeCnn
rm -rf status/TrainDeepfold1DForIcshapeBlstm
```

* Submit the jobs:
```bash
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshape
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeDense
bin/create_jobs.py CreateBsubJob -t TrainClassifierForIcshape -n 5
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeLogreg
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeCnn
bin/create_jobs.py CreateBsubJob -t TrainDeepfold1DForIcshapeBlstm
```
* Check the status of the jobs:
```bash
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshape
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeDense
bin/create_jobs.py CheckStatus -t TrainClassifierForIcshape
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeLogreg
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeCnn
bin/create_jobs.py CheckStatus -t TrainDeepfold1DForIcshapeBlstm
```
* Check for errors:
```bash
fgrep -r --color=auto Error logs/TrainDeepfold1DForIcshapeLogreg
```
## Evaluate the models
Clean old files:
```bash
rm -rf status/EvaluateDeepfold1DForIcshape
rm -rf status/EvaluateDeepfold1DForIcshapeDense
rm -rf status/EvaluateClassifierForIcshape
rm -rf status/EvaluateDeepfold1DForIcshapeLogreg
rm -rf status/EvaluateDeepfold1DForIcshapeCnn
rm -rf status/EvaluateDeepfold1DForIcshapeBlstm
```
Submit the jobs:
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshape
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeDense
bin/create_jobs.py CreateBsubJob -t EvaluateClassifierForIcshape
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeLogreg
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeCnn
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DForIcshapeBlstm
```
2. Check the status of the jobs:
```bash
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshape
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeDense
bin/create_jobs.py CheckStatus -t EvaluateClassifierForIcshape
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeLogreg
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeCnn
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DForIcshapeBlstm
```

## Report metrics and select models
```bash
bin/create_jobs.py CreateBsubJob -t MetricTable
rm -r status/MetricTable
bash jobs/MetricTable.sh
```
1. Report metrics:
```bash
bin/create_jobs.py CreateBsubJob -t MetricTable
for data_name in Lu_2016_invitro Lu_2016_invivo Lu_2016_invitro_hg38 Lu_2016_invivo_hg38 Spitale_2015_invivo Spitale_2015_invitro;do
    bin/report.py MetricTable --experiment-type icSHAPE --data-name $data_name \
        -o reports/MetricTable/icSHAPE/${data_name}.txt
    bin/report.py SelectModel --metric accuracy --metric-file reports/MetricTable/icSHAPE/${data_name}.txt \
        --num 3 -o selected_models/icSHAPE/${data_name}.json
done
```

## Basic statistics for Deepfold datasets

### Report the number of samples in each train/test dataset
```bash
for data_name in $(ls data/icSHAPE);do
    bin/report.py DeepfoldDatasetStatistics --experiment-type icSHAPE \
        --data-name ${data_name} -o reports/DeepfoldDatasetStatistics/icSHAPE/${data_name}/number_of_samples.txt
done
```

### Report the best metrics for each model
```bash
for data_name in $(ls data/icSHAPE);do
    for region in all 5UTR 3UTR CDS lncRNA;do
        bin/report.py CompareDeepfold1DMetrics \
            -i "reports/MetricTable/icSHAPE/d=${data_name},r=${region}.txt" \
            -o "reports/CompareDeepfold1DMetrics/icSHAPE/d=${data_name},r=${region}.txt"
    done
done
```

## Dense prediction
```bash
bin/deepfold2.py PredictDeepfold1D \
    -i data/Known/ct --format ct_dir \
    --model-file 'trained_models/icSHAPE/Lu_2016_invitro/r=CDS,p=5,w=100,m=resnet1_dense' \
    --swap-labels \
    --metric-by-sequence-file tmp/known.metric_by_sequence.resnet1_dense.txt
bin/deepfold2.py PredictDeepfold1D \
    -i data/Known/ct --format ct_dir \
    --model-file 'trained_models/icSHAPE/Lu_2016_invitro/r=CDS,p=5,w=100,m=resnet1' \
    --swap-labels \
    --metric-by-sequence-file tmp/known.metric_by_sequence.resnet1.txt
```


## Dense prediction using BUMHMM scores
```bash
bin/deepfold2.py PredictDeepfold1D \
    -i data/Known/ct --format ct_dir \
    --model-file 'trained_models/icSHAPE/Lu_2016_invitro.BUMHMM/r=CDS,p=5,w=100,m=resnet1_dense' \
    --swap-labels \
    --metric-by-sequence-file tmp/known.metric_by_sequence.BUMHMM.txt
```

## In silico mutate-and-map
```bash
bin/preprocess.py GenerateMutatedSequences \
    -i data/Known/fasta/5s_Bacillus-licheniformis-2.fa \
    -o tmp/5s_Bacillus-licheniformis-2.mutate_and_map.fa
bin/deepfold2.py PredictDeepfold1D \
    -i tmp/5s_Bacillus-licheniformis-2.mutate_and_map.fa --format fasta \
    --model-file 'trained_models/icSHAPE/Spitale_2015_invivo/r=CDS,p=5,w=100,m=resnet1_dense' \
    --swap-labels \
    --dense-pred-file tmp/5s_Bacillus-licheniformis-2.mutate_and_map.dense_predictions.h5
```
## Train separate models for each nucleotide
```bash
region=3UTR
bin/preprocess.py CreateDatasetFromGenomicData \
    -i data/icSHAPE/Lu_2016_invitro/${region} \
    --sequence-file ~/data/gtf/gencode.v19/sequences/${region}.transcript.fa \
    --window-size 100 \
    --stride 1 \
    --train-test-split 0.8 \
    --percentile 10 \
    --balance-nucleotide \
    --separate \
    -o tmp/Lu_2016_invitro.${region}.per_nuc
for nuc in A T C G;do
    echo "Training on nucleotide $nuc"
    bin/deepfold2.py TrainDeepfold1D \
        -i tmp/Lu_2016_invitro.${region}.per_nuc/$nuc \
        --model-script models/deepfold/logreg.py \
        --epochs 10 \
        --valid-file tmp/Lu_2016_invitro.${region}.per_nuc/A \
        --valid-xname X_test \
        --valid-yname y_test \
        --model-file tmp/Lu_2016_invitro.${region}.per_nuc.model/$nuc
done
for nuc in A T C G;do
    bin/report.py LogRegWeights \
        -i tmp/Lu_2016_invitro.${region}.per_nuc.model/$nuc \
        -o tmp/Lu_2016_invitro.${region}.per_nuc.weights/${nuc}.pdf
done
```
## Train the whole dataset after with nucleotide composition balanced
```bash
region=CDS
bin/preprocess.py CreateDatasetFromGenomicData \
    -i data/icSHAPE/Lu_2016_invitro/${region} \
    --sequence-file ~/data/gtf/gencode.v19/sequences/${region}.transcript.fa \
    --window-size 100 \
    --stride 1 \
    --train-test-split 0.8 \
    --percentile 10 \
    -o tmp/Lu_2016_invitro.${region}.balance_nucleotide \
    --balance-kmer \
    --kmer-start -1 --kmer-end 2 
bin/deepfold2.py TrainDeepfold1D \
    -i tmp/Lu_2016_invitro.${region}.balance_nucleotide \
    --model-script models/deepfold/logreg.py \
    --epochs 10 \
    --valid-file tmp/Lu_2016_invitro.${region}.balance_nucleotide \
    --valid-xname X_test \
    --valid-yname y_test \
    --model-file tmp/Lu_2016_invitro.${region}.balance_nucleotide.model
bin/report.py LogRegWeights \
    -i tmp/Lu_2016_invitro.${region}.balance_nucleotide.model \
    -o tmp/Lu_2016_invitro.${region}.balance_nucleotide.weights.pdf
bin/deepfold2.py PredictDeepfold1D \
    -i data/Known/ct --format ct_dir \
    --model-file tmp/Lu_2016_invitro.${region}.balance_nucleotide.model \
    --swap-labels \
    --metric-file tmp/Lu_2016_invitro.${region}.balance_nucleotide.metrics \
    --pred-file tmp/Lu_2016_invitro.${region}.balance_nucleotide.predictions
```

```bash
bin/deepfold2.py DrawModelOutputImages \
    -i 'trained_models/icSHAPE/Lu_2016_invitro/r=CDS,p=5,w=100,m=resnet1' \
    -o tmp/Lu_2016_invitro.resnet1.output_images 
```
## Predict calculated icSHAPE scores
```bash
bin/preprocess.py CreateRegressionDatasetForIcshape \
    -i data/icSHAPE/$data_name/$score_method/all \
    --feature scores \
    --window-size $window_size --stride 64 \
    --sequence-file /Share/home/shibinbin/data/gtf/gencode.${gencode_version}/sequences/${region}.transcript.fa \
    -o data/icSHAPE/$data_name/regression/r=${region},w=${window_size}
bin/deepfold2.py TrainDeepfold1D \
    -i data/icSHAPE/$data_name/regression/r=${region},w=${window_size} \
    --model-script models/regression/fcn1.py \
    --model-file trained_models/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1
bin/deepfold2.py TrainDeepfold1D \
    -i data/icSHAPE/$data_name/regression/r=${region},w=${window_size} \
    --model-script models/regression/mlp1.py \
    --model-file trained_models/icSHAPE/$data_name/r=${region},w=${window_size},m=mpl1,reg=1
bin/deepfold2.py EvaluateDeepfold1D \
    -i data/icSHAPE/$data_name/regression/r=${region},w=${window_size} \
    --model-file trained_models/icSHAPE/$data_name/r=${region},w=${window_size},m=mpl1,reg=1 \
    --metrics mean_squared_error,r2 \
    -o metrics/icSHAPE/$data_name/r=${region},w=${window_size},m=mpl1,reg=1
bin/deepfold2.py EvaluateDeepfold1D \
    -i data/icSHAPE/$data_name/regression/r=${region},w=${window_size} \
    --model-file trained_models/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1 \
    --metrics mean_squared_error,r2 \
    -o metrics/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1
```

## Evaulate on known structure using calculated icSHAPE scores
```bash
data_name=Spitale_2015_invitro
region=all
gencode_version=vM12
window_size=128
bin/deepfold2.py PredictDeepfold1D \
    -i data/Known/ct --format ct_dir \
    --metrics mean_squared_error,r2,roc_auc \
    --model-file trained_models/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1 \
    --metric-file metrics/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1 \
    --pred-file output/Known/icSHAPE/$data_name/r=${region},w=${window_size},m=fcn1,reg=1
```