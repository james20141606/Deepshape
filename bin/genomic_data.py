import h5py
import numpy as np
class GenomicData(object):
    def __init__(self, filename=None, feature_names=None, meta_names=None):
        """
        Arguments:
            filename: if not None, then load data from the file
            feature_names: a list of feature data to load
            meta_names: a list of metadata to load
        """
        self.feature_names = feature_names
        self.meta_names = meta_names
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        """Load data from an HDF5 file.
        At least three datasets named 'name', 'start', 'end' should be in the file and they should be
        1D array of the same length. A pair of ('start', 'end') is a "0-based, half-open" coordinate.
        Each 'feature' is an array of values of variable length associated with a name.
        Each 'meta' is a scalar associated with a name.
        """
        f = h5py.File(filename, 'r')
        for key in ['name', 'start', 'end']:
            if key not in f:
                raise KeyError('dataset name {} is not found in HDF5 file: {}'.format(key, self.filename))
        self.names = f['name'][:]
        self.namedict = dict(((name, i) for i, name in enumerate(self.names)))
        self.start = f['start'][:]
        self.end = f['end'][:]
        self.features = {}
        g = f['feature']
        if not self.feature_names:
            self.feature_names = f['feature'].keys()
        for name in self.feature_names:
            self.features[name] = g[name][:]
            setattr(self, name, self.features[name])
        self.meta = {}
        g = f['meta']
        if not self.meta_names:
            self.meta_names = f['meta'].keys()
        for name in self.meta_names:
            self.meta[name] = g[name][:]
            setattr(self, name, self.meta[name])
        f.close()

    def set_names(self, names):
        self.names = names
        self.namedict = dict(((name, i) for i, name in enumerate(self.names)))

    def feature(self, feature_name, name):
        if name not in self.namedict:
            return None
        else:
            i = self.namedict[name]
            if feature_name in self.features:
                return self.features[feature_name][self.start[i]:self.end[i]]
            else:
                return self.meta[feature_name][i]
    """
    def meta(self, meta_name, name):
        if name not in self.namedict:
            return None
        else:
            return self.meta[meta_name][self.namedict[name]]
    """
    def save(self, filename):
        f = h5py.File(filename, 'w')
        f.create_dataset('name', data=self.names)
        f.create_dataset('start', data=self.start)
        f.create_dataset('end', data=self.end)
        g = f.create_group('feature')
        for feature_name, values in self.features.iteritems():
            g.create_dataset(feature_name, data=self.features[feature_name])
        g = f.create_group('meta')
        for meta_name, values in self.meta.iteritems():
            g.create_dataset(meta_name, data=values)
        f.close()

    def __getitem__(self, key):
        if key in self.features:
            return self.features[key]
        if key in self.meta:
            return self.meta[key]
        return None

    @staticmethod
    def from_data(names, features={}, meta={}, create_namedict=True):
        """Create a GenomicData object from lists of names and values
        Arguments:
            names: an numpy array of names (dtype='S1')
            features: a dict of (feature_name, values) pairs where values is a list of numpy arrays.
                The number of items in values should be equal to the number of items in names.
            meta: a dict of (meta_name, values) pairs where values is a numpy array.
                The size of values should be equal to the size of names.
        """
        obj = GenomicData()
        obj.names = np.asarray(names, dtype='S')
        if create_namedict:
            obj.namedict = dict(((name, i) for i, name in enumerate(names)))
        obj.start = None
        obj.end = None
        obj.features = {}
        for feature_name, values in features.iteritems():
            if obj.start is None:
                length = np.asarray(map(len, values), dtype='int64')
                obj.end = np.cumsum(length)
                obj.start = obj.end - length
            if len(values) != len(names):
                raise ValueError('length of feature {} ({})is equal to the length of names ({})'.format(
                    feature_name, len(values), len(names)))
            obj.features[feature_name] = np.concatenate(values)
            setattr(obj, feature_name, obj.features[feature_name])
        obj.meta = {}
        for meta_name, values in meta.iteritems():
            if len(values) != len(names):
                raise ValueError('length of metadata {} ({}) is equal to the length of names ({})'.format(
                    meta_name, len(values), len(names)))
            obj.meta[meta_name] = values
            setattr(obj, meta_name, obj.meta[meta_name])
        return obj
