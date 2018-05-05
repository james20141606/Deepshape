## Preprocessing
```bash
bin/run_estimator.py MakeClassification -n 1000 -p 400 -o tmp/run_classification.h5
outdir=tmp/run_classification.svm.cv
[ -d $outdir ] || mkdir -p $outdir
cat > $outdir/param_grid.json <<PARAMSPEC
{
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "kernel": ["rbf"],
    "max_iter": [1000]
}
PARAMSPEC
bin/run_estimator.py CvPipeline --train-file tmp/run_classification.h5 \
    --test-file tmp/run_classification.h5 \
    --model-file tmp/run_classification.model \
    --n-jobs 8 \
    --outdir $outdir \
    --param-grid-file $outdir/param_grid.json --n-folds 3 \
    --metrics accuracy,roc_auc \
    --select-model-metric accuracy \
    --train-args ' --model-type sklearn --model-name svm' --execute

outdir=tmp/run_classification.random_forest.cv
[ -d $outdir ] || mkdir -p $outdir
cat > $outdir/param_grid.json <<PARAMSPEC
{
    "n_estimators": [10, 20, 30, 40, 50],
    "max_depth": [6]
}
PARAMSPEC
bin/run_estimator.py CvPipeline --train-file tmp/run_classification.h5 \
    --test-file tmp/run_classification.h5 \
    --model-file tmp/run_classification.model \
    --n-jobs 8 \
    --outdir $outdir \
    --param-grid-file $outdir/param_grid.json --n-folds 3 \
    --metrics accuracy,roc_auc \
    --select-model-metric accuracy \
    --train-args ' --model-type sklearn --model-name random_forest' --execute
```
