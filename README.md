# Vector-based Navigation Using Grid-like Representations in Artificial Agents

This repo serves as an open source implementation of the paper Vector-based Navigation Using Grid-like Representations in Artificial Agents from Banino et. al. at Google DeepMind. This paper can be found [here](https://doi.org/10.1038/s41586-018-0102-6).

This implementation utilizes [PyTorch](https://pytorch.org), and more specifically [namedtensor](https://github.com/harvardnlp/namedtensor), which names dimensions on tensors to make code more readable.

My goal in implementing this paper is to experiment with conditions and architectures that are variations of the ones implemented
by DeepMind to see where the emergence of Grid-Cell like properties emerge. 

So far I have implemented a precursory paper to generate simulated trajectories of rats in various environments as inspired by [Raudies et. al.](https://doi.org/10.1007/s10827-012-0396-6) and the proposed supervised agent from the DeepMind paper.

Next steps will include incorporating this model into the more complex RL agent using DeepMind lab and beginning experimentation.
