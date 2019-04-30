import os
from torch.utils.data import Dataset
import tensorflow as tf
import torch
import h5py
import collections

Feature = collections.namedtuple('Feature', ['name', 'type', 'data'])

class TFRecordWriter:
    def __init__(self, basepath, template=None, file_size=10000):
        self.basepath = basepath
        self.size = file_size

        if not template:
            template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)

        self.template = template

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

    def serialize_feature(self, data, t):
        if t == float:
            return self._float_feature(data)
        return data

    def write(self, features):
        """Generates lists of files for a given dataset version."""
        num_files = len(range(0, len(features[0].data), self.size))
        for i in range(num_files):
            fname = os.path.join(self.basepath, self.template.format(i, num_files - 1))
            print(f'Writing {self.size} records to file {fname}...')
            with tf.python_io.TFRecordWriter(fname) as writer:
                for j in range(i * self.size, (i + 1) * self.size):
                    feats = {f.name: self.serialize_feature(f.data[j], f.type) for f in features}
                    feats = tf.train.Features(feature=feats)
                    ex = tf.train.Example(features=feats).SerializeToString()
                    writer.write(ex)


class H5Dataset(Dataset):
    """Convenient interface for integrating PyTorch Dataset with H5Py"""

    def __init__(self, file_path, names):
        super(H5Dataset, self).__init__()

        # Store length and split size for later to avoid leaving file open
        with h5py.File(file_path) as h5_file:
            dset = h5_file[names[0]]
            self.len = dset.shape[0]
        self.file_path = file_path
        self.names = names
        self.file = None

    def __getitem__(self, index):
        # open file once, but in __getitem__ to allow for multiple workers
        if self.file is None:
            self.file = h5py.File(self.file_path, "r", swmr=True)
        return tuple(torch.tensor(self.file[n][index]) for n in self.names)

    def __len__(self):
        return self.len


class H5File:
    """File interface to load and save simulated trajectory data"""

    def __init__(self, fname, names=None):
        self.names = names
        if names is None:
            self.names = ['target_pos', 'target_hd', 'ego_vel', 'init_pos', 'init_hd']
        self.fname = fname
        self.empty = not self.exists()

    @property
    def attrs(self):
        with h5py.File(self.fname, "a") as f:
            return dict(f.attrs)

    def exists(self):
        with h5py.File(self.fname, "a") as f:
            return all(n in f for n in self.names)

    def save_data(self, datas):
        with h5py.File(self.fname, "a") as f:
            if not self.exists():
                print("Writing to new file...")
                for data, name in zip(datas, self.names):
                    dataset = f.create_dataset(
                        name,
                        data=data,
                        maxshape=(None, *data.shape[1:]),
                    )
                self.empty = False
            else:
                assert all(f[self.names[0]].shape[0] == f[n].shape[0] for n in self.names)
                print("Appending to existing file...")
                for data, name in zip(datas, self.names):
                    N = len(data)
                    dataset = f[name]
                    dataset.resize(dataset.shape[0] + N, axis=0)
                    dataset[-N:] = data

    def set_attrs(self, attrs):
        with h5py.File(self.fname, "a") as f:
            for k, v in attrs.items():
                f.attrs[k] = v

    def shrink_data(self, size):
        with h5py.File(self.fname, "a") as f:
            print("Slicing existing file...")
            for name in self.names:
                dataset = f[name]
                dataset.resize(size, axis=0)

    def to_dataset(self):
        # Convert data stored in file to a PyTorch dataset
        return H5Dataset(self.fname, self.names)

    def to_tfrecord(self, filename):
        writer = TFRecordWriter(filename)
        with h5py.File(self.fname, "a") as f:
            features = [Feature(name=name, data=f[name], type=float) for name in reversed(self.names)]
            writer.write(features)
