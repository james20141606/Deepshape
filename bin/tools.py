from workflow import Tool, InputFile, OutputFile

class CreateDatasetForIcshape(Tool):
    command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--percentile {percentile}
--window-size {window_size}
-o {outfile}'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size}'
    def generate_commands(self, params, task_name=None, command_only=False):
        if params.get('dense'):
            self.outputs['outfile'] = OutputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size},dense=1')
        if params.get('bumhmm') and params.get('dense'):
            self.command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--dense-output
--min-coverage 0.05
--cutoff1 0.4
--cutoff2 0.6
--window-size {window_size}
-o {outfile}'''
        elif params.get('bumhmm') and not params.get('dense'):
            self.command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--cutoff1 0.4
--cutoff2 0.6
--window-size {window_size}
-o {outfile}'''
        elif not params.get('bumhmm') and params.get('dense'):
            self.command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--dense-output
--min-coverage 0.05
--percentile {percentile}
--window-size {window_size}
-o {outfile}'''
        return super(self.__class__, self).generate_commands(params, task_name, command_only)
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/{region}.h5'),
            'sequence_file': InputFile('/Share/home/shibinbin/data/gtf/gencode.{gencode_version}/sequences/{region}.transcript.fa')}
        self.outputs = {'outfile': OutputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}')}

class CreateDatasetForIcshapeDense(Tool):
    command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--dense-output
--min-coverage 0.05
--percentile {percentile}
--window-size {window_size}
-o {outfile}'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/{region}'),
            'sequence_file': InputFile('/Share/home/shibinbin/data/gtf/gencode.{gencode_version}/sequences/{region}.transcript.fa')}
        self.outputs = {'outfile': OutputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size},dense=1')}

class CreateDatasetForIcshapeBumhmm(Tool):
    command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--cutoff1 0.4
--cutoff2 0.6
--window-size {window_size}
-o {outfile}'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/{region}'),
            'sequence_file': InputFile('/Share/home/shibinbin/data/gtf/gencode.{gencode_version}/sequences/{region}.transcript.fa')}
        self.outputs = {'outfile': OutputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}')}

class CreateDatasetForKnown(Tool):
    command = '''bin/preprocess.py CreateDatasetFromGenomicData
-i {infile}
--feature known
--sequence-file {sequence_file}
--stride 1
--train-test-split 0.8
--seed 24663
--cutoff1 0.5 --cutoff2 0.5
--window-size {window_size}
-o {outfile}'''
    unique_name = 'd={data_name},w={window_size}'
    def build(self):
        self.inputs = {'infile': InputFile('data/Known/{data_name}/known.h5'),
            'sequence_file': InputFile('data/Known/{data_name}/sequences.fa')}
        self.outputs = {'outfile': OutputFile('data/Known/{data_name}/deepfold/w={window_size}')}


class TrainDeepfold1DForKnown(Tool):
    command = '''bin/deepfold2.py TrainDeepfold1D
--infile data/Known/{data_name}/deepfold/w={window_size}
--model-script models/deepfold/{model_name}.py
--batch-size 100
--keras-log keras_log/Known/{data_name}/w={window_size},m={model_name}.csv
--keras-verbose 2
--model-file trained_models/Known/{data_name}/w={window_size},m={model_name}'''
    unique_name = 'd={data_name},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/Known/{data_name}/deepfold/w={window_size}')}
        self.outputs = {'model_file': OutputFile('trained_models/Known/{data_name}/w={window_size},m={model_name}')}

class TrainDeepfold1DForIcshape(Tool):
    command = '''bin/deepfold2.py TrainDeepfold1D
--infile {infile}
--model-script models/deepfold/{model_name}.py
--batch-size 100
--keras-log keras_log/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}.csv
--keras-verbose 2
--model-file {model_file}
--valid-file {infile}
--valid-xname X_test
--valid-yname y_test'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}')}
        self.outputs = {'model_file': OutputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}

class TrainDeepfold1DForIcshapeDense(TrainDeepfold1DForIcshape):
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size},dense=1')}
        self.outputs = {'model_file': OutputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}

class TrainClassifierForIcshape(Tool):
    command = '''bin/run_estimator.py CvPipeline
--train-file {infile}
--test-file {infile}
--model-file {model_file}
--outdir {cvdir}
--param-grid-file models/sklearn/param_grid.{model_name}.json
--n-folds 3
--n-jobs 5
--metrics accuracy,roc_auc
--select-model-metric accuracy
--flatten
--train-args ' --model-name {model_name}'
--execute'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}')}
        self.outputs = {'cvdir': OutputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}.cv'),
            'model_file': OutputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}

class EvaluateClassifierForIcshape(Tool):
    command = '''bin/run_estimator.py TestEstimator
-i {infile}
--model-type sklearn
--model-file {model_file}
--metrics accuracy,roc_auc,sensitivity,ppv
--flatten
-o {outfile}
'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}'),
            'model_file': InputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}
        self.outputs = {'outfile': OutputFile('metrics/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}

class EvaluateDeepfold1DForIcshape(Tool):
    command = '''bin/deepfold2.py EvaluateDeepfold1D
--infile {infile}
--model-file {model_file}
--metrics accuracy,roc_auc,sensitivity,ppv
--outfile {outfile}'''
    unique_name = 'd={data_name},r={region},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size}'),
            'model_file': InputFile('trained_models/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}
        self.outputs = {'outfile': OutputFile('metrics/icSHAPE/{data_name}/r={region},p={percentile},w={window_size},m={model_name}')}

class EvaluateDeepfold1DForIcshapeDense(EvaluateDeepfold1DForIcshape):
    def build(self):
        super(self.__class__, self).build()
        self.inputs['infile'] = InputFile('data/icSHAPE/{data_name}/deepfold/r={region},p={percentile},w={window_size},dense=1')

class EvaluateDeepfold1DCross(Tool):
    command = '''bin/deepfold2.py EvaluateDeepfold1D
--infile {infile}
--model-file {model_file}
--metrics accuracy,roc_auc,sensitivity,ppv
--outfile {outfile}'''
    unique_name = 'd={data_name},md={model_data_name},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile('data/{experiment_type}/{data_name}/deepfold/p={percentile},w={window_size}'),
            'model_file': InputFile('trained_models/{model_experiment_type}/{model_data_name}/p={percentile},w={window_size},m={model_name}')}
        self.outputs = {'outfile': OutputFile('metrics/cross/{experiment_type},{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}')}

class MetricTable(Tool):
    command = '''bin/report.py MetricTable
--experiment-type {experiment_type}
--data-name {data_name}
--region {region}
-o {outfile}'''
    unique_name = 'e={experiment_type},d={data_name},r={region}'
    def build(self):
        self.inputs = {}
        self.outputs = {'outfile': OutputFile('reports/MetricTable/{experiment_type}/d={data_name},r={region}.txt')}

class BaseDistributionForIcshape(Tool):
    command = '''bin/preprocess.py BaseDistribution
--feature icshape
--percentile 5
--score-file {score_file}
--sequence-file {sequence_file}
--outfile {outfile}'''
    unique_name = 'd={data_name}'
    def build(self):
        self.inputs = {'score_file': InputFile('data/icSHAPE/{data_name}/icshape.h5'),
            'sequence_file': InputFile()}
        self.outputs = {'outfile': InputFile('reports/IcshapeBaseDistribution/{data_name}.pdf')}

class MaxExpect(Tool):
    command = '''if [ -z "$RNASTRUCTURE_PATH" ];then echo "Error: variable RNASTRUCTURE_PATH is empty";exit 1;fi;
export DATAPATH=$RNASTRUCTURE_PATH/data_tables/;
true $(mkdir -p $(dirname {pfsfile}));
$RNASTRUCTURE_PATH/exe/partition {infile} {pfsfile};
$RNASTRUCTURE_PATH/exe/MaxExpect {pfsfile} {outfile};
rm {pfsfile}'''
    unique_name = 'd={data_name},md={model_data_name},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'infile': InputFile()}
        self.outputs = {'outdir': OutputFile(),
            'pfsfile': OutputFile()}

class RME(Tool):
    command = '''true $(mkdir -p {outdir});
$RME_PATH/bin/RME {infile} {outdir} -p 1 -m {m}
--gamma1 {gamma1} --gamma2 {gamma2}'''
    def build(self):
        self.inputs = {'infile': InputFile()}
        self.outputs = {'outdir': OutputFile()}

class Fold(Tool):
    command = '''outdir=\$(dirname {outfile});
true $(mkdir -p $(dirname {outfile}));
DATAPATH=/Share/home/shibinbin/apps/src/RNAstructure/data_tables/ /Share/home/shibinbin/apps/src/RNAstructure/exe/Fold {infile} {outfile}'''
    def build(self):
        self.inputs = {'infile': InputFile()}
        self.outputs = {'outdir': OutputFile()}

class ScoreStructure(Tool):
    command = '''
bin/report.py ScoreStructure --true-file data/Known/ct/ \
    --pred-file {indir} \
    -o {outfile}
'''
    def build(self):
        self.inputs = {'pred_dir': InputFile()}
        self.outputs = {'outfile': InputFile()}

class PredictDeepfold1DOnKnown(Tool):
    command = '''bin/deepfold2.py PredictDeepfold1D
--model-file {model_file}
--infile {infile} --format ct --swap-labels
--split
--fillna 0.5
--outfile {outdir}
'''
    unique_name = 'd={data_name},md={model_data_name},p={percentile},w={window_size},m={model_name}'
    def build(self):
        self.inputs = {'model_file': InputFile('trained_models/{model_experiment_type}/{model_data_name}/p={percentile},w={window_size},m={model_name}.h5'),
            'infile': InputFile('data/Known/ct')}
        self.outputs = {'outdir': OutputFile('output/deepfold/Known,{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}')}
