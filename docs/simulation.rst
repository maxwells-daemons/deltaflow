simulation
==========

.. automodule:: simulation

High-level API
--------------

The high-level simulation API consists of two functions:
 - :func:`simulation.simulate` -- run a multi-timestep simulation starting from some initial conditions.
 - :func:`simulation.step` -- advance a simulation by a single step, given the state at the last step.

Both are differentiable with respect to their vector field inputs, letting you backpropagate through the whole simulation process.

The high-level API is available through :code:`deltaflow.*` as well as :code:`deltaflow.simulation.*`.

.. autofunction:: simulation.simulate

.. autofunction:: simulation.step

.. autoclass:: simulation.SimulationConfig

Low-level simulation functions
------------------------------

This API exposes the internals used for individual simulation components.
Use at your own risk!

.. autofunction:: simulation._get_predecessor_coordinates

.. autofunction:: simulation._advect

.. autofunction:: simulation._divergence_2d

.. autofunction:: simulation._compute_pressure

.. autofunction:: simulation._diffuse
