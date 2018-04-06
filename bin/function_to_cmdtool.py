#! /usr/bin/env python
import inspect
import sys, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate code for a CommandLineTool from a python function')
    parser.add_argument('function', type=str,
        help='function name (may include package and module names)')
    parser.add_argument('--class-name', '-n', type=str, default='Tool',
        help='class name')
    parser.add_argument('--tabs', '-t', action='store_true',
        help='use tabs rather than 4 spaces')
    args = parser.parse_args()

    packages = args.function.split('.')
    if len(packages) > 1:
        exec 'from %s import %s as func'%('.'.join(packages[:-1]), packages[-1])
    else:
        exec 'import %s as func'%args.function
    argspec = inspect.getargspec(func)
    if args.tabs:
        tabs = '\t'
    else:
        tabs = '    '
    fout = sys.stdout
    fout.write('class %s(CommandLineTool):\n'%args.class_name)
    fout.write(tabs + 'description = %s\n'%repr(args.function))
    fout.write(tabs + 'arguments = [\n')
    iarg = 0
    for arg, defval in zip(argspec.args, argspec.defaults):
        if iarg > 0:
            fout.write(',\n')
        fout.write(tabs + tabs + "Argument('%s'"%arg)
        if isinstance(defval, bool):
            if not defval:
                fout.write(", action='store_true'")
            else:
                fout.write(", action='store_false'")
        elif isinstance(defval, int):
            fout.write(', type=int')
        elif isinstance(defval, float):
            fout.write(', type=float')
        else:
            fout.write(', type=str')
        if isinstance(defval, (int, str, float)):
            fout.write(", default=%s"%repr(defval))
        fout.write(')')
        iarg += 1
    fout.write('\n')
    fout.write(tabs + ']\n')
    fout.write(tabs + 'def __call__(self):\n')
    fout.write(tabs + tabs + 'pass\n')
