from torch.utils.data import Dataset
import torch
import h5py


class H5Dataset(Dataset):
    """Convenient interface for integrating PyTorch Dataset with H5Py"""

    def __init__(self, file_path, dataset):
        super(H5Dataset, self).__init__()

        # Store length and split size for later to avoid leaving file open
        with h5py.File(file_path) as h5_file:
            dset = h5_file.get(dataset)
            self.len = dset.shape[0]
            self.split_size = [
                2,
                dset.attrs["place_cell_count"],
                dset.attrs["hd_cell_count"],
                3,
            ]

        self.file_path = file_path
        self.dataset = dataset
        self.file = None

    def __getitem__(self, index):
        # open file once, but in __getitem__ to allow for multiple workers
        if self.file is None:
            self.file = h5py.File(self.file_path, "r", swmr=True)[self.dataset]
        xs, cs, hs, inp = torch.split(
            torch.tensor(self.file[index]),
            split_size_or_sections=self.split_size,
            dim=1,
        )
        return inp, cs, hs, xs

    def __len__(self):
        return self.len


class H5File:
    """File interface to load and save simulated trajectory data"""

    def __init__(self, fname, dataset):
        self.fname = fname
        self.dataset = dataset
        self.empty = not self.exists()

    @property
    def attrs(self):
        with h5py.File(self.fname, "a") as f:
            dataset = f[self.dataset]
            return dict(dataset.attrs)

    def exists(self):
        with h5py.File(self.fname, "a") as f:
            return self.dataset in f

    def save_data(self, data):
        dataset_name = self.dataset
        with h5py.File(self.fname, "a") as f:
            if not self.exists():
                print("Writing to new file...")
                dataset = f.create_dataset(
                    dataset_name,
                    data=data,
                    maxshape=(None, data.shape[1], data.shape[2]),
                )
                self.empty = False
            else:
                N = len(data)
                print("Appending to existing file...")
                dataset = f[dataset_name]
                dataset.resize(dataset.shape[0] + N, axis=0)
                dataset[-N:] = data

    def set_attrs(self, attrs):
        with h5py.File(self.fname, "a") as f:
            dataset = f[self.dataset]
            for k, v in attrs.items():
                dataset.attrs[k] = v

    def shrink_data(self, size):
        with h5py.File(self.fname, "a") as f:
            print("Slicing existing file...")
            dataset = f[self.dataset]
            dataset.resize(size, axis=0)

    def to_dataset(self):
        # Convert data stored in file to a PyTorch dataset
        return H5Dataset(self.fname, self.dataset)
