Welcome to DeltaFlow's documentation!
=====================================

.. image:: ../resources/delta_fluid.png
    :width: 400
    :alt: A delta symbol formed out of fluid
    :align: center

DeltaFlow is a GPU-accelerated differentiable fluid simulator written in JAX.

By taking derivatives through the entire simulation, you can optimize simulation inputs with gradient descent. For example, you could find initial velocities that carefully orchestrate fluid particles to form an image. Take that, entropy!

Since it's end-to-end JAX, it's composable with other differentiable JAX components. For example, you could use `Haiku <https://github.com/deepmind/dm-haiku>`_ to train a neural network to interact with the fluid simulator, and since the environment is fully differentiable you wouldn't need reinforcement learning.

Also, it's fast. On a recent GPU, DeltaFlow can run a 4K simulation at 40 FPS. Smaller simulations often achieve hundreds of frames per second, even when computing gradients.

Examples
--------

Check out interactive examples with a free cloud GPU through Colaboratory:
 - `Animating fluids <https://colab.research.google.com/drive/1iOVCJJ5qYQYOP6Ui81E8gtmo3UBmRLQk?usp=sharing>`_
 - `Backpropagating through simulation <https://colab.research.google.com/drive/1Q2wiE_ros0-WQtJMGW0w_Hbl1PHau0hV?usp=sharing>`_

The example notebooks are also available on `GitHub <https://github.com/maxwells-daemons/deltaflow/tree/main/examples>`_.

Contents
--------

.. toctree::
   :maxdepth: 2

   simulation
   utils


Installation
------------

For a CPU-only installation, clone `the repository <https://github.com/maxwells-daemons/deltaflow>`_, then :code:`pip install .`.

For GPU support, you'll need to first install the version of :code:`jaxlib` appropriate for your python, CUDA, and CuDNN versions.
See `the jaxlib install instructions <https://github.com/google/jax#pip-installation>`_ for more details.


Code
----

Check out the code `on GitHub <https://github.com/maxwells-daemons/deltaflow>`_.


License
-------

This code is licensed under the `MIT License <https://github.com/maxwells-daemons/deltaflow/blob/main/LICENSE>`_.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
