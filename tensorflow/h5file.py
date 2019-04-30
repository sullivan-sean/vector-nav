import h5py


class Saver:
    def __init__(self, fname):
        self.file = h5py.File(fname, "a")

    def save_traj(self, traj):
        datasets = ['init_pos', 'init_hd', 'ego_vel', 'target_pos', 'target_hd']
        f = self.file
        for t, d in zip(traj, datasets):
            try:
                dataset = f[d]
                dataset.resize(dataset.shape[0] + t.shape[0], axis=0)
                dataset[-t.shape[0]:] = t
            except Exception:
                f.create_dataset(d, data=t, maxshape=(None, *t.shape[1:]))
