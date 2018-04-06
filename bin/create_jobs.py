#! /usr/bin/env python
from cmdtool import CommandLineTool, Argument
import itertools
import string
import sys, os, copy
from collections import OrderedDict
from ioutils import make_dir
from workflow import Task, ParamGrid, InputFile, OutputFile
import tools

"""
class CreateDatasetForIcshape(Task):
    def build(self):
        self.paramlist = ParamGrid({
            'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo',
                'Lu_2016_invitro_published', 'Lu_2016_invivo_published'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'gencode_version': ['v19']
        }).to_list()
        self.paramlist += ParamGrid({
            'data_name': ['Lu_2016_invitro_hg38', 'Lu_2016_invivo_hg38'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'gencode_version': ['v26']
        }).to_list()
        self.paramlist += ParamGrid({
            'data_name': ['Spitale_2015_invivo', 'Spitale_2015_invitro'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'gencode_version': ['vM12']
        }).to_list()
        self.tool = tools.CreateDatasetForIcshape()
"""
class CreateDatasetForIcshape(Task):
    def build(self):
        self.paramlist = ParamGrid({
            'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo'],
            'percentile': [5],
            'window_size': [100],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'gencode_version': ['v19']
        }).to_list()
        self.paramlist += ParamGrid({
            'data_name': ['Spitale_2015_invivo', 'Spitale_2015_invitro'],
            'percentile': [5],
            'window_size': [100],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'gencode_version': ['vM12']
        }).to_list()
        paramlist_bumhmm = []
        for params in self.paramlist:
            params_new = copy.copy(params)
            params_new['data_name'] = params_new['data_name'] + '.BUMHMM'
            params_new['bumhmm'] = True
            paramlist_bumhmm.append(params_new)
        self.paramlist += paramlist_bumhmm
        self.tool = tools.CreateDatasetForIcshape()

class CreateDatasetForIcshapeDense(CreateDatasetForIcshape):
    def build(self):
        super(CreateDatasetForIcshapeDense, self).build()
        paramlist_dense = []
        for params in self.paramlist:
            params_new = copy.copy(params)
            params_new['dense'] = True
            paramlist_dense.append(params_new)
        self.paramlist = paramlist_dense
        self.tool = tools.CreateDatasetForIcshape()

class TrainDeepfold1DForIcshape(CreateDatasetForIcshape):
    def build(self):
        super(TrainDeepfold1DForIcshape, self).build()
        paramlist = []
        for params in self.paramlist:
            if params['region'] == 'all':
                continue
            for model_name in ['logreg', 'mlp1', 'conv1', 'resnet1']:
                params_new = copy.copy(params)
                params_new['model_name'] = model_name
                paramlist.append(params_new)
        self.paramlist = paramlist
        self.tool = tools.TrainDeepfold1DForIcshape()

class TrainDeepfold1DForIcshapeDense(CreateDatasetForIcshapeDense):
    def build(self):
        super(TrainDeepfold1DForIcshapeDense, self).build()
        paramlist = []
        for params in self.paramlist:
            if params['region'] == 'all':
                continue
            for model_name in ['logreg_dense', 'mlp1_dense', 'conv1_dense', 'resnet1_dense']:
                params_new = copy.copy(params)
                params_new['model_name'] = model_name
                paramlist.append(params_new)
        self.paramlist = paramlist
        self.tool = tools.TrainDeepfold1DForIcshapeDense()

class TrainClassifierForIcshape(Task):
    def build(self):
        self.paramlist = ParamGrid({
            'data_name': ['Lu_2016_invivo', 'Lu_2016_invitro',
                'Spitale_2015_invivo', 'Spitale_2015_invitro'],
            'percentile': [5],
            'window_size': [100],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA'],
            'model_name': ['svm', 'random_forest']
            }).to_list()
        self.tool = tools.TrainClassifierForIcshape()

class TrainDeepfold1DForIcshapeLogreg(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invivo', 'Lu_2016_invitro',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Lu_2016_invitro_hg38', 'Lu_2016_invivo_hg38',
            'Spitale_2015_invivo', 'Spitale_2015_invitro'],
            'percentile': [5],
            'window_size': [25, 50, 75, 100, 125, 150, 175, 200],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA', 'all'],
            'model_name': ['logreg']
        }).to_list()
        self.tool = tools.TrainDeepfold1DForIcshape()

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

class EvaluateClassifierForIcshape(TrainClassifierForIcshape):
    def build(self):
        super(EvaluateClassifierForIcshape, self).build()
        self.tool = tools.EvaluateClassifierForIcshape()

class EvaluateDeepfold1DForIcshape(TrainDeepfold1DForIcshape):
    def build(self):
        super(EvaluateDeepfold1DForIcshape, self).build()
        self.tool = tools.EvaluateDeepfold1DForIcshape()

class EvaluateDeepfold1DForIcshapeDense(TrainDeepfold1DForIcshapeDense):
    def build(self):
        super(EvaluateDeepfold1DForIcshapeDense, self).build()
        self.tool = tools.EvaluateDeepfold1DForIcshapeDense()

class EvaluateDeepfold1DForIcshapeLogreg(Task):
    def build(self):
        self.paramlist = TrainDeepfold1DForIcshapeLogreg().paramlist
        self.tool = tools.EvaluateDeepfold1DForIcshape()

class EvaluateDeepfold1DForIcshapeCnn(Task):
    def build(self):
        self.paramlist = TrainDeepfold1DForIcshapeCnn().paramlist
        self.tool = tools.EvaluateDeepfold1DForIcshape()

class EvaluateDeepfold1DForIcshapeBlstm(Task):
    def build(self):
        self.paramlist = TrainDeepfold1DForIcshapeBlstm().paramlist
        self.tool = tools.EvaluateDeepfold1DForIcshape()

class MetricTable(Task):
    def build(self):
        self.paramlist = ParamGrid({'experiment_type': ['icSHAPE'],
            'data_name': ['Lu_2016_invivo', 'Lu_2016_invitro',
                    'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
                    'Lu_2016_invivo_hg38', 'Lu_2016_invitro_hg38',
                    'Spitale_2015_invivo', 'Spitale_2015_invitro',
                    'Lu_2016_invivo.BUMHMM', 'Lu_2016_invitro.BUMHMM',
                    'Spitale_2015_invivo.BUMHMM', 'Spitale_2015_invitro.BUMHMM'],
            'region': ['CDS', '5UTR', '3UTR', 'lncRNA']
        }).to_list()
        self.tool = tools.MetricTable()

class BaseDistributionForIcshapeHuman(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro', 'Lu_2016_invivo']}).to_list()
        self.tool = BaseDistributionForIcshape()
        self.tool.inputs['sequence_file'] = InputFile('~/data/genomes/fasta/Human/hg19.transcript.v19.fa')

class BaseDistributionForIcshapeHumanPublished(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Lu_2016_invitro_published', 'Lu_2016_invivo_published']}).to_list()
        self.tool = BaseDistributionForIcshape()
        self.tool.inputs['sequence_file'] = InputFile('~/data/genomes/fasta/Human/hg19.transcript.v19.noversion.fa')

class BaseDistributionForIcshapeMouse(Task):
    def build(self):
        self.paramlist = ParamGrid({'data_name': ['Spitale_2015_invitro', 'Spitale_2015_invivo'],
            'sequence_file': ['~/data/genomes/fasta/Mouse/mm10.transcript.vM12.fa']
        }).to_list()
        self.tool = BaseDistributionForIcshape()

class EvaluateFoldOnKnown(Task):
    def build(self):
        sequence_dir = 'data/Known/fasta'
        sequence_names = map(lambda x: os.path.splitext(x)[0], os.listdir(sequence_dir))
        self.paramlist = ParamGrid({'data_name': ['All'],
            'sequence_name': sequence_names}).to_list()
        self.tool = Fold()
        self.tool.unique_name = 'd={data_name},s={sequence_name}'
        self.tool.inputs['infile'] = InputFile('data/Known/fasta/{sequence_name}.fa')
        self.tool.outputs['outfile'] = OutputFile('output/Fold/Known/{sequence_name}.ct')

class EvaluateMaxExpectOnKnown(Task):
    def build(self):
        sequence_dir = 'data/Known/fasta'
        sequence_names = map(lambda x: os.path.splitext(x)[0], os.listdir(sequence_dir))
        self.paramlist = ParamGrid({'data_name': ['All'],
            'sequence_name': sequence_names}).to_list()
        self.tool = MaxExpect()
        self.tool.unique_name = 'd={data_name},s={sequence_name}'
        self.tool.inputs['infile'] = InputFile('data/Known/fasta/{sequence_name}.fa')
        self.tool.outputs['pfsfile'] = OutputFile('output/MaxExpect/Known/{sequence_name}.pfs')
        self.tool.outputs['outfile'] = OutputFile('output/MaxExpect/Known/{sequence_name}.ct')

class EvaluateRMEOnKnown(Task):
    def build(self):
        sequence_names = open('data/Known/names.txt').read().split()
        self.paramlist = []
        for model_data_name in ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo']:
            model_paramlist = ParamFile('selected_models/icSHAPE/{}.json'.format(model_data_name)).to_list()

            for params in model_paramlist:
                if params['window_size'] >= 160:
                    continue
                params['model_data_name'] = params['data_name']
                params['model_experiment_type'] = 'icSHAPE'
                params['data_name'] = 'All'
                params['experiment_type'] = 'Known'
                params['m'] = 0.1
                params['gamma1'] = 0.1
                params['gamma2'] = 0.1
                for name in sequence_names:
                    params_seq = dict(params)
                    params_seq['sequence_name'] = name
                    self.paramlist.append(params_seq)
        self.tool = RME()
        self.tool.unique_name = 'd={data_name},md={model_data_name},p={percentile},w={window_size},m={model_name},s={sequence_name}'
        self.tool.inputs['infile'] = InputFile('output/deepfold/Known,{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}/{sequence_name}')
        self.tool.outputs['outdir'] = OutputFile('output/RME/Known,{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}')

class ScoreStructureRMEOnKnown(EvaluateRMEOnKnown):
    def build(self):
        self.paramlist = []
        for model_data_name in ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo']:
            model_paramlist = ParamFile('selected_models/icSHAPE/{}.json'.format(model_data_name)).to_list()
            for params in model_paramlist:
                if params['window_size'] >= 160:
                    continue
                params['model_data_name'] = params['data_name']
                params['model_experiment_type'] = 'icSHAPE'
                params['data_name'] = 'All'
                params['experiment_type'] = 'Known'
                self.paramlist.append(params)
        self.tool = ScoreStructure()
        self.tool.unique_name = 'd={data_name},md={model_data_name},p={percentile},w={window_size},m={model_name}'
        self.tool.inputs['indir'] = InputFile('output/RME/Known,{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}')
        self.tool.outputs['outfile'] = OutputFile('reports/StructurePredictionMetrics/RME/Known,{data_name}/{model_experiment_type},{model_data_name}/p={percentile},w={window_size},m={model_name}.txt')

class PredictDeepfold1DIcshapeOnKnown(Task):
    def build(self):
        self.paramlist = []
        for model_data_name in ['Lu_2016_invitro', 'Lu_2016_invivo',
            'Lu_2016_invitro_published', 'Lu_2016_invivo_published',
            'Spitale_2015_invitro', 'Spitale_2015_invivo']:
            model_params = ParamFile('selected_models/icSHAPE/{}.json'.format(model_data_name)).to_list()
            for params in model_params:
                params['model_data_name'] = params['data_name']
                params['model_experiment_type'] = 'icSHAPE'
                params['data_name'] = 'All'
                params['experiment_type'] = 'Known'
                self.paramlist.append(params)
        self.tool = PredictDeepfold1DOnKnown()

class CreateBsubJob(CommandLineTool):
    description = 'A tool for creating batch jobs'
    arguments = [Argument('task_name', short_opt='-t', type=str, required=True,
        choices = Task.get_all_task_names(), help='task name'),
        Argument('outfile', short_opt='-o', type=str),
        Argument('param_file', type=str, help='parameter file in JSON format which is decoded as a list of dict'),
        Argument('logdir', type=str, default='logs'),
        Argument('jobdir', type=str, default='jobs'),
        Argument('taskdir', type=str, default='tasks'),
        Argument('job_name', type=str),
        Argument('max_jobs', short_opt='-j', type=int, default=60, help='maximum number of tasks in an array job'),
        Argument('threads', short_opt='-n', type=int, default=1),
        Argument('queue', type=str, default='Z-LU'),
        Argument('shell', type=str, default='/bin/bash'),
        Argument('rc_pre', type=str),
        Argument('rc_post', type=str)]
    def __call__(self):
        template_bsub ='''#!{shell}
#BSUB -J "{job_name}[1-{max_jobs}]"
#BSUB -oo {logdir}/{job_name}/stdout.%I.log
#BSUB -eo {logdir}/{job_name}/stderr.%I.log
#BSUB -R "span[hosts=1]"
#BSUB -n {threads}
#BSUB -q {queue}

# get number of tasks and task rank
export LSB_JOBINDEX_END=${{LSB_JOBINDEX_END:=1}}
export LSB_JOBINDEX=${{LSB_JOBINDEX:=1}}
NUM_TASKS=$LSB_JOBINDEX_END
RANK=$LSB_JOBINDEX

# run commands from a command list file
# only commands that belongs to the task will be run

run_tasks(){{
    sed '/^\s*$/ d' | sed "$RANK~$NUM_TASKS !d" \\
        | xargs -d '\\n' -t -l $@ -I '{{}}' bash -c '{{}}'
}}

run_tasks <<TASKFILES
'''
        task = Task.get_task(self.task_name)()
        if self.param_file is not None:
            import json
            with open(self.param_file, 'r') as f:
                task.paramlist = json.load(f)

        fout = sys.stdout
        if not self.outfile:
            self.outfile = os.path.join(self.jobdir, task.__class__.__name__ + '.sh')
        self.logger.info('generate bsub script: {}'.format(self.outfile))
        make_dir(self.jobdir)
        fout = open(self.outfile, 'w')
        if not self.job_name:
            self.job_name = task.__class__.__name__
        n_commands = len(task.paramlist)
        self.max_jobs = min(self.max_jobs, n_commands)
        fout.write(template_bsub.format(**vars(self)))
        for cmd in task.generate_commands():
            if self.rc_pre:
                fout.write('if [ -f {} ];then . {};fi; '.format(self.rc_pre))
            fout.write(cmd.replace('$', '\$'))
            if self.rc_post:
                fout.write('if [ -f {} ];then . {};fi')
            fout.write('\n')
        fout.write('TASKFILES\n')
        fout.close()
        make_dir(self.logdir + '/' + self.job_name)

class PrintCommands(CommandLineTool):
    arguments = [Argument('task_name', short_opt='-t', type=str, required=True,
        choices = Task.get_all_task_names(), help='task name'),
        Argument('param_file', type=str, help='parameter file in JSON format which is decoded as a list of dict')
        ]
    def __call__(self):
        task = Task.get_task(self.task_name)()
        if self.param_file is not None:
            import json
            with open(self.param_file, 'r') as f:
                task.paramlist = json.load(f)
        for cmd in task.generate_commands():
            print cmd

class CleanStatus(CommandLineTool):
    arguments = [Argument('task_name', short_opt='-t', type=str, required=True,
        choices=Task.get_all_task_names(), help='task name'),
        Argument('status_dir', type=str, default='status')]
    def __call__(self):
        task = Task.get_task(self.task_name)()
        for params in task.paramlist:
            unique_name = task.tool.unique_name.format(**params)
            print 'rm ' + os.path.join(self.status_dir, task.__class__.__name__, unique_name)

class CheckStatus(CommandLineTool):
    arguments = [Argument('task_name', short_opt='-t', type=str, required=True,
        choices=Task.get_all_task_names(), help='task name'),
        Argument('summary', action='store_true'),
        Argument('monitor', action='store_true'),
        Argument('interval', type=float, default=2, help='watch interval'),
        Argument('status_dir', type=str, default='status')]
    def check_status(self, task):
        if self.summary:
            finished = 0
            for params in task.paramlist:
                unique_name = task.tool.unique_name.format(**params)
                if os.path.exists(os.path.join(self.status_dir, task.__class__.__name__, unique_name)):
                    finished += 1
            sys.stdout.write('\r\bFinished: {}/{} ({:.2f}%)'.format(
                finished, len(task.paramlist), 100*float(finished)/len(task.paramlist)))
            sys.stdout.flush()
        else:
            for params in task.paramlist:
                unique_name = task.tool.unique_name.format(**params)
                if os.path.exists(os.path.join(self.status_dir, task.__class__.__name__, unique_name)):
                    status = '\x1B[32mYES\x1B[0m'
                else:
                    status = '\x1B[31mNO\x1B[0m'
                print '{}({})\t{}'.format(task.__class__.__name__, unique_name, status)
    def __call__(self):
        task = Task.get_task(self.task_name)()
        if self.monitor:
            import time
            while True:
                self.check_status(task)
                time.sleep(self.interval)
        else:
            self.check_status(task)
        sys.stdout.write('\n')


class CheckReady(CommandLineTool):
    arguments = [Argument('task_name', short_opt='-t', type=str, required=True,
        choices=Task.get_all_task_names(), help='task name')]
    def __call__(self):
        task = Task.get_task(self.task_name)
        for params in task.paramlist:
            unique_name = task.unique_name.format(**params)
            ready, missing = task.ready(params)
            if ready:
                print '{}({})\t\x1B[32mYES\x1B[0m'.format(task.__name__, unique_name)
            else:
                print '{}({})\t\x1B[31mNO\x1B[0m\t{}'.format(task.__name__, unique_name, missing)


class PrintTaskTree(CommandLineTool):
    def print_task_tree(self, task_class, level=0):
        print ''.join(['    ']*level) + task_class.__name__
        for c in task_class.__subclasses__():
            self.print_task_tree(c, level + 1)
    def __call__(self):
        self.print_task_tree(Task, 0)

if __name__ == '__main__':
    try:
        CommandLineTool.from_argv()()
    except KeyboardInterrupt:
        pass
