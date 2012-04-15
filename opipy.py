#!/usr/bin/env python

"""
Open anything in IPython
"""


def start_ipython(ns, argv=[]):
    import IPython

    try:
        embedshell = IPython.Shell.IPShellEmbed(argv=argv)
    except AttributeError:
        IPython.embed(user_ns=ns)  # 0.12
    else:
        embedshell(local_ns=ns)  # 0.10


class OpenInIPythonBase(object):

    cmdname = None
    exts = []

    def get_ns(self, *args, **kwds):
        raise NotImplementedError

    def print_ns(self, ns):
        from pprint import pprint
        pprint(ns)

    def run(self, *args, **kwds):
        ipy = kwds.pop('ipy', [])
        ns = self.get_ns(*args, **kwds)
        self.print_ns(ns)
        start_ipython(ns=ns, argv=ipy)

    def connect_subparser(self, subpersers):
        parser = self.add_parser(subpersers.add_parser(self.cmdname))
        parser.set_defaults(func=self.run)  # used via `applyargs`
        return parser

    def add_parser(self, parser):
        parser.add_argument(
            'filepath', nargs='+',
            help=('path to the files to open (extension: {0})'
                  .format(self.exts)))
        return parser


def applyargs(func, **kwds):
    return func(**kwds)


# Define sub-commands:

class OpenJSON(OpenInIPythonBase):

    cmdname = 'json'
    exts = ['json', 'js']

    def get_ns(self, filepath):
        import json
        ns = {'filepath': filepath, 'json': json}
        data = ns['data'] = {}
        return ns

        datasfmt = 'data{0}'.format
        ns.update(('file{0}'.format(i), f) for (i, f) in enumerate(filepath))

        for (i, path) in enumerate(filepath):
            data[path] = ns[datasfmt(i)] = json.load(file(path))

        return ns


class OpenPickle(OpenInIPythonBase):

    cmdname = 'pickle'
    exts = ['pickle']

    @staticmethod
    def import_module(module):
        import sys
        __import__(module)
        return sys.modules[module]

    @classmethod
    def from_import(cls, module, names):
        mod = cls.import_module(module)
        return dict((n, getattr(mod, n)) for n in names)

    @staticmethod
    def parse_module_path(module_path):
        """
        Parse a string into the argument for `from_import` function

        >>> parse_module_path("pylookup")
        ('pylookup', [])
        >>> parse_module_path("pylookup:Element")
        ('pylookup', ['Element'])
        >>> parse_module_path("pylookup:Element,update")
        ('pylookup', ['Element', 'update'])

        """
        aslist = module_path.split(':', 1)
        if len(aslist) == 1:
            return (aslist[0], [])
        else:
            return (aslist[0], aslist[1].split(','))

    def get_ns(self, filepath, mod=[]):
        try:
            import cPickle as pickle
        except:
            import pickle
        ns = {'pickle': pickle, 'filepath': filepath}
        data = ns['data'] = {}
        datasfmt = 'data{0}'.format
        ns.update(('file{0}'.format(i), f) for (i, f) in enumerate(filepath))

        # this hack is a workaround of AttributeError
        for modpath in mod:
            globals().update(
                self.from_import(*self.parse_module_path(modpath)))

        for (i, path) in enumerate(filepath):
            objlist = data[path] = ns[datasfmt(i)] = []
            with open(path) as f:
                try:
                    while True:
                        objlist.append(pickle.load(f))
                except EOFError:
                    pass

        return ns

    def print_ns(self, ns):
        print "loaded:"
        print ', '.join(sorted(ns))
        return ns

    def add_parser(self, parser):
        super(OpenPickle, self).add_parser(parser)
        parser.add_argument(
            '-m', '--mod', default=[], action='append',
            help='"MODULE[:NAME[,NAME[,...]]]" to specify object to load')
        return parser


class OpenReST(OpenInIPythonBase):

    cmdname = 'rst'
    exts = ['rst', 'rest', 'txt']

    def get_ns(self, filepath):
        import docutils
        from docutils.core import publish_doctree
        from docutils import nodes

        doctree = [publish_doctree(source=file(f).read()) for f in filepath]

        ns = {
            'filepath': filepath,
            'doctree': doctree,
            'nodes': nodes,
            'docutils': docutils,
            }

        return ns

    def print_ns(self, ns):
        print "loaded:"
        print ', '.join(sorted(ns))


class OpenMercurial(OpenInIPythonBase):

    cmdname = 'hg'

    def get_ns(self, path):
        import mercurial.hg as hg
        import mercurial.ui
        ui = mercurial.ui.ui()
        repo = hg.repository(ui, path=path)
        ns = {
            'hg': hg,
            'ui': ui,
            'repo': repo,
            }

        return ns

    def add_parser(self, parser):
        parser.add_argument(
            'path', nargs='?', default='.',
            help='a path to Mercurial repository')
        return parser


class OpenNumpy(OpenInIPythonBase):

    cmdname = 'numpy'
    exts = ['npy', 'npz']

    def get_ns(self, filepath, mixin=False):
        import numpy
        ns = {}
        data = map(numpy.load, filepath)
        ns.update(numpy=numpy, filepath=filepath, data=data)
        ns.update(('data_{0}'.format(i), d) for (i, d) in enumerate(data))
        if mixin:
            ns.update(
                (k, d[k])
                for d in data if isinstance(d, numpy.lib.npyio.NpzFile)
                for k in d.keys()
                )
        return ns

    def print_ns(self, ns):
        print "loaded:"
        print ', '.join(sorted(ns))

    def add_parser(self, parser):
        super(OpenNumpy, self).add_parser(parser)
        parser.add_argument(
            '-m', '--mixin', action='store_true',
            help='mix all contents in *.npz file')
        return parser


class OpenPandasHDF5(OpenInIPythonBase):

    cmdname = 'pandas-hdf5'
    exts = ['hdf5']

    def get_ns(self, filepath, mode='r'):
        import pandas
        ns = {}

        def openhdf5(p):
            return pandas.HDFStore(p, mode=mode)

        data = map(openhdf5, filepath)
        ns.update(pandas=pandas, filepath=filepath, data=data)
        ns.update(('data_{0}'.format(i), d) for (i, d) in enumerate(data))
        return ns

    def add_parser(self, parser):
        super(OpenPandasHDF5, self).add_parser(parser)
        parser.add_argument(
            '-m', '--mmode', default='r',
            help='mode (a/w/r/r+) to open the file')
        return parser


class OpenNEO(OpenInIPythonBase):

    cmdname = 'neo'

    def get_ns(self, filepath):
        import neo

        def getreader(path):
            for io in neo.io.iolist:
                for ext in io.extensions:
                    if path.endswith(ext):
                        return io(path)

        readerlist = map(getreader, filepath)
        ns = {'neo': neo, 'readerlist': readerlist, 'filepath': filepath}
        if readerlist:
            ns['reader'] = readerlist[0]
            ns['filename'] = filepath[0]

        return ns

    def print_ns(self, ns):
        print "loaded:"
        print ', '.join(sorted(ns))


# Register classes

CLASS_LIST = [
    OpenJSON,
    OpenPickle,
    OpenReST,
    OpenMercurial,
    OpenNumpy,
    OpenPandasHDF5,
    OpenNEO,
]


def get_parser(class_list):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--ipy', default='',
        help='arguments to path to ipython. '
        '(e.g.: --ipy="-pdb -colors Linux")')
    subpersers = parser.add_subparsers()
    for method in class_list:
        method().connect_subparser(subpersers)
    return parser


def main():
    parser = get_parser(CLASS_LIST)
    args = parser.parse_args()
    applyargs(**vars(args))


if __name__ == '__main__':
    main()
