# Vector-based Navigation Using Grid-like Representations in Artificial Agents

This repo serves as an open source implementation of the paper Vector-based Navigation Using Grid-like Representations in Artificial Agents from Banino et. al. at Google DeepMind. This paper can be found [here](https://doi.org/10.1038/s41586-018-0102-6).

This implementation utilizes [PyTorch](https://pytorch.org), and more specifically [namedtensor](https://github.com/harvardnlp/namedtensor), which names dimensions on tensors to make code more readable.

My goal in implementing this paper is to experiment with conditions and architectures that are variations of the ones implemented
by DeepMind to see where the emergence of Grid-Cell like properties emerge. 

So far I have implemented a precursory paper to generate simulated trajectories of rats in various environments as inspired by [Raudies et. al.](https://doi.org/10.1007/s10827-012-0396-6) and the proposed supervised agent from the DeepMind paper.

Next steps will include incorporating this model into the more complex RL agent using DeepMind lab and beginning experimentation.

## Play around with code

To play around with the code here, you can clone the repo and install the requirements

```
git clone https://github.com/sullivan-sean/vector-nav.git
cd vector-nav
pip install -r requirements.txt
```
Training the pytorch model can be done as follows. First you can generate rat trajectories as below:

```python
from scene import SquareCage
from trajectory_simulator import TrajectorySimulator

scene = SquareCage(2.2)
simulator = TrajectorySimulator(scene)
simulator.plot_trajectories(N=5)
```

Other shapes of cages can be found in `scene.py` and you will find many
parameters that are tunable on the simulator as well, including the angular
and forward velocities of the rats.

To save trajectories to an hdf5 file and load for training you can use the
following in addition to the above:

```python
from dataloader import H5File
from torch.utils.data import DataLoader

file = H5File('<filename>.hdf5', '<dataset_name>')
simulator.save(N=1000)

dataset = file.to_dataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
```

Finally, you can build and train a model with:

```python
from model import VecNavModel
from train import train_model

VNM = VecNavModel(12, 256, g_size=512).cuda()
losses = train_model(VNM, dataloader, lr=1e-5, num_epochs=40)
```

The PyTorch model does _not_ produce grid cells as the tensorflow model does. This is a problem I have been struggling to solve since the sixth week of the semester. To try and remedy this problem, I implemented the model from scratch 3 times, played around with various trajectory motion models, transferred Deepmind's trajectories to a PyTorch readable format, transferred my trajectories to a tensorflow compatible format, altered initialization and regularization, and even copied weights from the tensorflow model directly into the PyTorch model to ensure identical intialization. At its core, the problem seems to be either human error on my part, or a difference deep within in PyTorch and Tensorflow and their optimization strategies. The results in my paper come from the modified tensorflow model, which takes after the released code from Deepmind.
