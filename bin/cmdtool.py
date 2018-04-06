from __future__ import print_function
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import sys
import os

class Argument(object):
    def __init__(self, name, type=str,
            short_opt=None, long_opt=None, required=False, default=None,
            delimiter=',', action='store', metavar=None, nargs=None,
            choices=None, help=None):
        self.name = name
        self._type = type
        self.short_opt = short_opt
        if long_opt:
            self.long_opt = long_opt
        else:
            self.long_opt = '--' + name.replace('_', '-')
        self.default = default
        self.required = required
        self.help = help
        self.delimiter = delimiter
        self.action = action
        self.nargs = nargs
        self.choices = choices
        if (metavar is None) and (choices is None):
            self.metavar = str(self._type.__name__).upper()
        else:
            self.metavar = metavar

    @property
    def type(self):
        if self._type == list:
            return str
        elif self.action == 'store_true':
            return None
        else:
            return self._type

    def get_value(self, value):
        if (value is not None) and (self._type == list):
            return value.split(self.delimiter)
        else:
            return value

class CommandLineTool(object):
    description = 'A command line tool'
    arguments = []
    def __init__(self, **kwargs):
        self.args = {}
        for arg in self.arguments:
            if arg.required and (kwargs.get(arg.name) is None):
                raise ValueError('argument {} is required for {}'.format(arg.name, self.__class__.__name__))
            setattr(self, arg.name, kwargs[arg.name])
            self.args[arg.name] = kwargs[arg.name]
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_argv(cls, argv=sys.argv):
        """Parse command line arguments and store the parsed values as attributes
        """
        commands = dict(((c.__name__, c) for c in CommandLineTool.__subclasses__()))
        if len(argv) < 2:
            print('Usage: {} command [options]'.format(sys.argv[0]), file=sys.stderr)
            print('Available commands: ' + ' '.join(commands.keys()), file=sys.stderr)
            sys.exit(1)
        if argv[1] not in commands:
            print('Error: unknown command: {}'.format(argv[1]), file=sys.stderr)
            print('Available commands: ' + ' '.join(commands.keys()), file=sys.stderr)
            sys.exit(1)
        c = commands[argv[1]]
        parser = argparse.ArgumentParser(argv[0] + ' ' + argv[1],
            description=c.description)
        for arg in c.arguments:
            opts = [arg.long_opt]
            if arg.short_opt is not None:
                opts.append(arg.short_opt)
            if arg.action in ('store_true', 'store_false'):
                parser.add_argument(*opts, dest=arg.name,
                    default=arg.default, action=arg.action, help=arg.help)
            else:
                parser.add_argument(*opts, dest=arg.name, nargs=arg.nargs,
                    type=arg.type, required=arg.required, metavar=arg.metavar,
                    default=arg.default, action=arg.action, choices=arg.choices,
                    help=arg.help)
        parser.add_argument('--verbose', action='store_true',
            help='show more information (set logging level to DEBUG)')
        if len(argv) <= 2:
            argv = []
        else:
            argv = argv[2:]
        args = parser.parse_args(argv)
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        parsed_args = {}
        for arg in c.arguments:
            value = getattr(args, arg.name)
            parsed_args[arg.name] = arg.get_value(value)
        else:
            return c(**parsed_args)

    def __call__(self):
        raise NotImplementedError('abstract class CommandLineTool cannot be used directly')
