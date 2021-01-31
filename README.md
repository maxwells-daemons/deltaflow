<p align="center">
  <img src="./resources/delta_fluid.png" />
</p>

# DeltaFlow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q2wiE_ros0-WQtJMGW0w_Hbl1PHau0hV?usp=sharing)
[![Documentation Status](https://readthedocs.org/projects/deltaflow/badge/?version=latest)](https://deltaflow.readthedocs.io/en/latest/?badge=latest)
![License shield](https://img.shields.io/github/license/maxwells-daemons/deltaflow)

DeltaFlow is a GPU-accelerated differentiable fluid simulator written in JAX.

By taking derivatives through the entire simulation, you can optimize simulation inputs with gradient descent.
For example, you could find initial velocities that carefully orchestrate fluid particles to form an image.
Take that, entropy!

Since it's end-to-end JAX, it's composable with other differentiable JAX components.
For example, you could use [Haiku](https://github.com/deepmind/dm-haiku) to train a neural network to interact with the fluid simulator,
and since the environment is fully differentiable you wouldn't need reinforcement learning.

Also, it's fast. On a recent GPU, DeltaFlow can run a 4K simulation at 40 FPS.
Smaller simulations often achieve hundreds of frames per second, even when computing gradients.

## Usage

The main API is two functions: `simulate()`, which runs a fluid simulation, and `step()`, which runs a single step.
Both are differentiable with respect to their vector field inputs.

For detailed usage examples, see the notebooks under `examples/`.

## Documentation

Documentation is hosted on [Read the Docs](https://deltaflow.readthedocs.io/en/latest/index.html).

## Installation

For a CPU-only installation, clone the repository, then `pip install .`.

For GPU support, you'll need to first install the version of `jaxlib` appropriate for
your python, CUDA, and CuDNN versions. See the
[jaxlib install instructions](https://github.com/google/jax#pip-installation)
for more details.
