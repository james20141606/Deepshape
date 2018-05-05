```bash
bin/analyze_base_density.py BaseDensityCorrelationBetweenDatasets \
    -a output/icSHAPE_preprocess/Lu_2016_invitro/background.normalized.rt \
    -b output/icSHAPE_preprocess/Lu_2016_invivo/background.normalized.rt \
    -o reports/BaseDensityCorrelationBetweenDatasets/icSHAPE/Lu_2016_invitro.Lu_2016_invivo
bin/analyze_base_density.py CreateRegressDatasetForBaseDensity \
    -i output/icSHAPE_preprocess/Lu_2016_invitro/background.normalized.rt \
    --sequence-file ~/data/genomes/fasta/Human/hg19.transcript.v19.fa \
    --window-size 100 \
    --smooth \
    --max-samples 200000 \
    -o data/icSHAPE_base_density/Lu_2016_invitro/w=100.h5


bin/run_estimator.py CreateCvIndex \
    -i data/icSHAPE_base_density/Lu_2016_invitro/w=100.h5 \
    -k 3 -o trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/cv_index.h5
bin/run_estimator.py TrainEstimator --regress \
    --cv-index-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/cv_index.h5 \
    --cv-fold 0 \
    --scale-targets \
    --hyperparam '{"window_size": 100}' \
    --model-type keras --model-script models/regression/conv1.py --model-name 'conv1' \
    -i data/icSHAPE_base_density/Lu_2016_invitro/w=100.h5 \
    --model-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/0/0.model \
    --valid-metric-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/0/0.metric.h5
```
```bash
bin/run_estimator.py TrainEstimator \
    --regress --flatten \
    --cv-index-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/cv_index.h5 --cv-fold 0 \
    --model-type sklearn --model-name linear_regression \
    -i data/icSHAPE_base_density/Lu_2016_invitro/w=100.h5 \
    --model-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/0/0.model \
    --valid-metric-file trained_models/icSHAPE_base_density/Lu_2016_invitro/w=100.cv/0/0.metric.h5
```
## Test the classification models
```bash
prefix=tmp/classification
bin/run_estimator.py MakeClassification \
    -n 10000 -p 100 --n-informative 40 --n-redundant 20 -o ${prefix}.h5
bin/run_estimator.py CreateCvIndex \
    -i ${prefix}.h5 -k 3 -o ${prefix}.cv/cv_index.h5
bin/run_estimator.py HyperParamGrid \
    --grid-spec '{"kernel": ["rbf"], "C": [1.0], "max_iter": [1000], "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] }' \
    -o ${prefix}.cv/hyperparam_list.txt
hyperparam_index=0
while read hyperparam;do
    for cv_fold in $(seq 0 2);do
        bin/run_estimator.py TrainEstimator \
            --cv-index-file ${prefix}.cv/cv_index.h5 --cv-fold $cv_fold \
            --hyperparam "$hyperparam" \
            --model-type sklearn --model-name svm \
            -i ${prefix}.h5 \
            --model-file ${prefix}.cv/$hyperparam_index/${cv_fold}.model \
            --valid-metric-file ${prefix}.cv/$hyperparam_index/${cv_fold}.metric.h5
    done
    hyperparam_index=$(($hyperparam_index + 1))
done < ${prefix}.cv/hyperparam_list.txt
```

## Test the regression models
```bash
prefix=tmp/regression
bin/run_estimator.py MakeRegression \
    --scale-targets -n 10000 -p 100 --bias 1 -o ${prefix}.h5
bin/run_estimator.py CreateCvIndex \
    -i ${prefix}.h5 -k 3 -o ${prefix}.cv/cv_index.h5
bin/run_estimator.py TrainEstimator --regress \
    --cv-index-file ${prefix}.cv/cv_index.h5 --cv-fold 0 \
    --hyperparam '{"kernel": "rbf", "gamma": 0.1, "C": 1.0}' \
    --model-type sklearn --model-name svr \
    -i ${prefix}.h5 \
    --model-file ${prefix}.cv/0/0.model \
    --valid-metric-file ${prefix}.cv/0/0.metric.h5
bin/run_estimator.py TrainEstimator --regress \
    --cv-index-file tmp/regression.cv/cv_index.h5 --cv-fold 0 \
    --model-type sklearn --model-name linear_regression \
    -i ${prefix}.h5 \
    --model-file ${prefix}.cv/0/0.model \
    --valid-metric-file ${prefix}.cv/0/0.metric.h5
bin/run_estimator.py TrainEstimator --regress \
    --cv-index-file ${prefix}.cv/cv_index.h5 --cv-fold 0 \
    --model-type keras --model-script models/regression/mlp1.py --model-name mlp1 \
    --hyperparam '{"n_features": 100}' \
    -i ${prefix}.h5 \
    --model-file ${prefix}.cv/0/0.model \
    --valid-metric-file ${prefix}.cv/0/0.metric.h5

```
