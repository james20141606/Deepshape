# Evaluate Deepfold on Known RNA structures

## Evaluate Deepfold on known structures
1. Create a task in bin/create_jobs.py
```python
class EvaluateDeepfold1DIcshapeOnKnown(Task):
    def build(self):
        self.paramlist = []
        for data_name in ['Lu_2016_invivo', 'Lu_2016_invitro',
                'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
                'Spitale_2015_invivo', 'Spitale_2015_invitro']:
            paramlist = ParamFile('selected_models/icSHAPE/{}.json'.format(data_name)).to_list()
            for params in paramlist:
                params['experiment_type'] = 'Known'
                params['model_experiment_type'] = 'icSHAPE'
                params['data_name'] = 'All'
                params['model_data_name'] = data_name
            self.paramlist += paramlist
        self.tool = EvaluateDeepfold1DCross()
        self.tool.command += ' --swap-labels'
        self.tool.inputs['infile'] = InputFile('data/{experiment_type}/{data_name}/deepfold/w={window_size}.h5')
```
2. Submit the job
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateDeepfold1DIcshapeOnKnown -n 2
bsub < jobs/EvaluateDeepfold1DIcshapeOnKnown.sh
```
3. Check the status
```bash
bin/create_jobs.py CheckStatus -t EvaluateDeepfold1DIcshapeOnKnown
```
4. Generate a metric table
```bash
bin/report.py MetricTableCross --experiment-type Known --data-name All \
    --outfile reports/MetricTableCross/Known,All.txt
```

## Compare Deepfold 1D profile with known structures
```bash
for d in $(ls "output/deepfold/Known,All/");do
    for model_param in $(ls "output/deepfold/Known,All/$d");do
        bin/report.py CompareDeepfold1DWithKnown \
            --infile "output/deepfold/Known,All/$d/$model_param" \
            --outfile "reports/CompareDeepfold1DWithKnown/Known,All/$d/${model_param}.pdf"
    done
done
```
## Run RME with deepfold prediction as restraints
1. Predict 1D structure profile from CT files
```bash
bin/deepfold2.py PredictDeepfold1D
    --model-file "trained_models/icSHAPE/Lu_2016_invitro/p=5,w=40,m=logreg.h5" \
    --infile data/Known/ct --format ct --swap-labels --outfile tmp/Known.txt
```
Create a task:
```python

```

```bash
bin/create_jobs.py CreateBsubJob -t PredictDeepfold1DIcshapeOnKnown
bsub < jobs/PredictDeepfold1DIcshapeOnKnown.sh
```
Issue: sequence shorter than window size cannot be predicted

## Evaluate Fold on known structures
1. Create a task
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateFoldOnKnown
bsub < jobs/EvaluateFoldOnKnown.sh
```
2. Score the structures
```bash
bin/report.py ScoreStructure --true-file data/Known/ct/ \
    --pred-file output/Fold/Known/ \
    -o reports/StructurePredictionMetrics/Fold/Known.txt
```
## Evaluate MaxExpect on known structures
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateMaxExpectOnKnown
bsub < jobs/EvaluateMaxExpectOnKnown.sh
```
Score the structures
```bash
bin/report.py ScoreStructure --true-file data/Known/ct/ \
    --pred-file output/MaxExpect/Known/ \
    -o reports/StructurePredictionMetrics/MaxExpect/Known.txt
```

## Evaluate RME on known structure
```bash
bin/create_jobs.py CreateBsubJob -t EvaluateRMEOnKnown
bin/create_jobs.py CheckStatus -t EvaluateRMEOnKnown
```
Score the structures
```bash
bin/create_jobs.py CreateBsubJob -t ScoreStructureRMEOnKnown
```
## Compare RME with MaxExpect
```bash
for d in $(ls reports/StructurePredictionMetrics/RME/Known,All/);do
    model_data_name="$d"
    for f in $(ls "reports/StructurePredictionMetrics/RME/Known,All/$d/");do
        model_param=$(basename "$f" .txt)
        for metric in sensitivity ppv;do
        bin/report.py CompareStructurePredictionMetrics \
            --infile1 reports/StructurePredictionMetrics/MaxExpect/Known.txt \
            --infile2 "reports/StructurePredictionMetrics/RME/Known,All/$d/$f" \
            --metric ${metric} \
            --outfile "reports/CompareStructurePredictionMetrics/MaxExpect/RME/Known,All/$d/${model_param}.${metric}.pdf" \
            --title "Differnce of {metric} between MaxExpect and RME ($model_data_name)"$'\n'"(model: ${model_param})"$'\n'"(mean: {mean}, median: {median})"
        done
    done
done
```

## Evaluate icSHAPE data on known structure
The results can be found in the RNAex benchmark directory:
`/lustre/users/shibinbin/Projects/RNAex`.
