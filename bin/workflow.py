import itertools
import string
import sys, os

class Template(str):
    def eval(self, params=None):
        if params is not None:
            return self.format(**params)
        else:
            return self

class InputFile(object):
    def __init__(self, path=''):
        self.path = Template(path)

    def ready(self, params=None):
        return os.path.isfile(self.path.eval(params))

    def eval(self, params=None):
        return self.path.eval(params)

class OutputFile(object):
    def __init__(self, path=''):
        self.path = Template(path)

    def ready(self, params=None):
        return os.path.isdir(os.path.dirname(self.path.eval(params)))

    def eval(self, params=None):
        return self.path.eval(params)

class Directory(object):
    def __init__(self, name, path):
        self.name = name
        self.path = Template(path)

class ParamGrid(object):
    def __init__(self, grid=None, **kwargs):
        if grid is not None:
            self.grid = grid
        else:
            self.grid = kwargs

    def get_params(self):
        names, value_grid = zip(*self.grid.items())
        for values in itertools.product(*value_grid):
            yield dict(zip(names, values))

    def to_list(self):
        return list(self.get_params())

    def __iter__(self):
        return self.get_params()

class ParamList(object):
    def __init__(varlist, **kwargs):
        if varlist is not None:
            self.varlist = varlist
        else:
            self.varlist = kwargs

    def get_params(self):
        names = self.varlist.keys()
        for values in zip(self.varlist.values()):
            yield dict(zip(names, values))

class ParamFile(object):
    def __init__(self, filename):
        import json
        with open(filename, 'r') as f:
            self.varlist = json.load(f)

    def to_list(self):
        return self.varlist

status_dir = 'status'
class Tool(object):
    def __init__(self):
        self.arguments = OrderedDict()
        self.build()

    def generate_commands(self, params, task_name=None, command_only=False):
        prolog = '''if [ -f "{status_dir}/{task_name}/{unique_name}" ];
then echo "Task {task_name}/{unique_name} has finished."; exit 0; fi;
set -e'''.replace('\n', ' ')
        epilog = '''[ -d "{status_dir}/{task_name}" ] || mkdir -p "{status_dir}/{task_name}";
touch "{status_dir}/{task_name}/{unique_name}";'''.replace('\n', ' ')
        command = self.command.replace('\n', ' ')
        params['unique_name'] = self.unique_name.format(**params)
        params['status_dir'] = status_dir
        params['task_name'] = task_name
        for a in self.inputs:
            params[a] = self.inputs[a].eval(params)
        for a in self.outputs:
            params[a] = self.outputs[a].eval(params)
        unique_name = self.unique_name.format(**params)
        return '; '.join([prolog.format(**params),
            command.format(**params),
            epilog.format(**params)])

    def build(self):
        pass

class Task(object):
    def __init__(self):
        self.build()

    def ready(self, params=None):
        missing = []
        for req in self.tool.inputs:
            if not req.ready(params):
                missing.append('{}({})'.format(req.__class__.__name__, req.eval(params)))
        if len(missing) > 0:
            return False, missing
        return True, missing

    @classmethod
    def _all_subclasses(cls, subclasses=[]):
        for c in cls.__subclasses__():
            c._all_subclasses(subclasses)
        subclasses.append((cls.__name__, cls))

    @classmethod
    def all_subclasses(cls):
        subclasses = []
        cls._all_subclasses(subclasses)
        return dict(subclasses)

    @staticmethod
    def get_all_task_names():
        return Task.all_subclasses().keys()

    @staticmethod
    def get_task(task_name):
        #all_tasks = dict((c.__name__, c) for c in Task.__subclasses__())
        all_tasks = Task.all_subclasses()
        if task_name not in all_tasks:
            raise KeyError('unknown task name: {}'.format(task_name))
        return all_tasks[task_name]

    def build(self):
        raise NotImplementedError('abstract class Task should not be used directly')

    def generate_commands(self):
        self.build()
        for params in self.paramlist:
            yield self.tool.generate_commands(params, task_name=self.__class__.__name__)
