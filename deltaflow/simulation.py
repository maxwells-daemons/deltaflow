"""
An API for differentiable fluid simulation.

Units are normalized to grid squares per unit time.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import tqdm


# TODO: add control over boundary conditions
class SimulationConfig(NamedTuple):
    """
    A configuration object determining a simulation's physical properties.

    Parameters
    ----------
    delta_t
        The time elapsed in each timestep.
        Smaller values produce more accurate results but advance the simulation slower.
    density_coeff
        A coefficient on fluid density.
        Higher values cause fluids to respond slower to pressure gradients.
        Values too far from 1.0 may produce unrealistic effects.
    diffusion_coeff
        A coefficient on diffusion rate and viscosity.
        Higher values cause faster diffusion and greater viscosity; values should be
        very small. A value of 0.0 simulates an inviscid flow and simulates faster.
    pressure_iterations
        The number of iterations used to compute pressure.
        Larger values produce more accurate results but simulate slower.
    """

    delta_t: float = 0.05
    density_coeff: float = 1.0
    diffusion_coeff: float = 1e-3
    pressure_iterations: int = 16


def _get_predecessor_coordinates(velocity: jnp.ndarray, delta_t: float) -> jnp.ndarray:
    """
    For each point on the grid, get the fractional coordinates of a particle that
    would move to the center of that gridsquare at the next timestep.

    Parameters
    ----------
    velocity
        The fluid velocity field. Shape: [y, x, y/x].
    delta_t
        The time elapsed in each timestep.

    Returns
    -------
    jnp.ndarray
        For each grid square, the (fractional) coordinates of the grid square that
        would move to its center at the next timestep. Shape: [y/x, y, x].
    """
    # Compile-time: precompute coordinate grid
    grid_coords = np.mgrid[: velocity.shape[0], : velocity.shape[1]]

    # Move each grid index by -velocity
    return grid_coords - velocity.transpose((2, 0, 1)) * delta_t


@partial(jax.vmap, in_axes=(2, None), out_axes=2)
def _advect(field: jnp.ndarray, predecessor_coords: jnp.ndarray) -> jnp.ndarray:
    """
    Transport a vector field by reading values moving into the center of each square.

    Parameters
    ----------
    field
        The vector field to transport. Shape: [y, x, any].
    predecessor_coords
        The predecessor coordinates computed from the velocity. Shape: [y/x, y, x].

    Returns
    -------
    jnp.ndarray
        The field, advected by one timestep.
    """
    return jax.scipy.ndimage.map_coordinates(field, predecessor_coords, order=1)


def _divergence_2d(field: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the divergence of a 2D vector field.

    Parameters
    ----------
    field
        Any 2D vector field. Shape: [y, x, 2].

    Returns
    -------
    jnp.ndarray
        The divergence of `field` as a scalar field. Shape: [y, x].
    """
    return jnp.gradient(field[:, :, 0], axis=0) + jnp.gradient(field[:, :, 1], axis=1)


def _compute_pressure(
    advected_velocity: jnp.ndarray, pressure: jnp.ndarray, pressure_iterations: int
) -> jnp.ndarray:
    """
    Compute the (unitless) pressure of a fluid.

    Uses Jacobi iteration, initialized with the last frame's pressure.

    Parameters
    ----------
    advected_velocity
        The fluid's velocity field for this frame, advected but without
        pressure correction. Shape: [y, x, 2].
    pressure
        Last frame's pressure field, used as an initial guess for the pressure solver.
        If no such estimate is available (e.g. for the first frame), a zero field may
        be passed. Shape: [y, x].
    pressure_iterations
        The number of iterations used to compute pressure.
        Must be static during JIT tracing.

    Returns
    -------
    jnp.ndarray
        The estimated pressure field for the fluid at this frame. Shape: [y, x].
    """
    # Compile-time: precompute surround kernel
    surround_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    surround_kernel = jax.device_put(surround_kernel)

    velocity_divergence = _divergence_2d(advected_velocity)

    for i in range(pressure_iterations):
        pressure = (
            jax.scipy.signal.convolve2d(pressure, surround_kernel, mode="same")
            - velocity_divergence
        ) / 4

    return pressure


@partial(jax.vmap, in_axes=(2, None, None), out_axes=2)
def _diffuse(field: jnp.ndarray, diffusion_coeff: float, delta_t: float) -> jnp.ndarray:
    """
    Average each value in a vector field closer to its neighbors to simulate
    diffusion and viscosity.

    Parameters
    ----------
    field
        The vector field to diffuse. Shape: [y, x, any].
    diffusion_coeff
        A coefficient determining the amount of diffusion at each frame.
        Must be static during JIT tracing.
    delta_t
        The time elapsed in each timestep. Must be static during JIT tracing.

    Returns
    -------
    jnp.ndarray
        `field`, with diffusion applied for this frame.
    """
    # Compile-time: precompute neighbor averaging kernel
    neighbor_weight = diffusion_coeff * delta_t
    neighbor_kernel = np.array(
        [
            [0, neighbor_weight / 4, 0],
            [neighbor_weight / 4, 1 - 4 * neighbor_weight, neighbor_weight / 4],
            [0, neighbor_weight / 4, 0],
        ]
    )
    neighbor_kernel = jax.device_put(neighbor_kernel)

    return jax.scipy.signal.convolve2d(field, neighbor_kernel, mode="same")


@partial(jax.jit, static_argnums=4)
def step(
    color: jnp.ndarray,
    velocity: jnp.ndarray,
    force: jnp.ndarray,
    pressure: jnp.ndarray,
    config: SimulationConfig = SimulationConfig(),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Advance the simulation by a single timestep.

    Supports differentiation through all arguments except `config`.
    All fields must match in their spatial dimensions.

    This function is JITted, since compiling multiple steps together doesn't yield much
    speedup. Using a different spatial resolution or `config` will trigger
    recompilation; doing this often will slow down simulation dramatically.

    Parameters
    ----------
    color
        The RGB color field of the fluid at the last frame. Shape: [y, x, 3].
    velocity
        The velocity field of the fluid at the last frame. Shape: [y, x, y/x].
    force
        A static force field to apply at this frame. Shape: [y, x, y/x].
    pressure
        The pressure field of the fluid at the last frame. Shape: [y, x].
        If this was the first frame, zeroes may be passed instead.
    config
        The simulation's physical configuration.
        Statically traced, so changing this argument triggers recompilation.
        Defaults to a reasonable default configuration.

    Returns
    -------
    color : jnp.ndarray
        The RGB color field of the fluid at the next frame. Shape: [y, x, 3].
    velocity : jnp.ndarray
        The velocity field of the fluid at the next frame. Shape: [y, x, y/x].
    pressure : jnp.ndarray
        The pressure field of the fluid at the next frame. Shape: [y, x].
    """
    # Advection: fluid particles move to their new locations
    predecessor_coords = _get_predecessor_coordinates(velocity, config.delta_t)
    color = _advect(color, predecessor_coords)
    velocity = _advect(velocity, predecessor_coords)

    # Apply external forces
    velocity = velocity + force * config.delta_t

    # Apply pressure gradient force
    pressure = _compute_pressure(velocity, pressure, config.pressure_iterations)
    pressure_gradient = jnp.stack(jnp.gradient(pressure), 2)
    velocity = velocity - pressure_gradient / config.density_coeff

    # Diffusion and viscosity
    if config.diffusion_coeff > 0.0:
        color = _diffuse(color, config.diffusion_coeff, config.delta_t)
        velocity = _diffuse(velocity, config.diffusion_coeff, config.delta_t)

    return color, velocity, pressure


def simulate(
    timesteps: int,
    color: jnp.ndarray,
    velocity: Optional[jnp.ndarray] = None,
    force: Optional[jnp.ndarray] = None,
    config: SimulationConfig = SimulationConfig(),
    return_frames: bool = True,
    disable_progress_bar: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run a multi-timestep fluid simulation.

    The value of `return_frames` determines what this function does:
      - If True (default), every frame's color and velocity are returned.
        This enables animation.
      - If False, only the last frame's color and velocity are returned.
        This is faster and supports computing gradients through the whole simulation.

    All fields must match in their spatial dimensions.

    Parameters
    ----------
    timesteps
        The number of timesteps to run the simulation for.
    color
        The RGB color field of the fluid at the first frame. Shape: [y, x, 3].
    velocity
        The velocity field of the fluid at the first frame. Shape: [y, x, y/x].
        If None (default), initialize a zero velocity field.
    force
        A static force field to apply at each frame. Shape: [y, x, y/x].
        If None (default), initialize a zero force field.
    config
        The simulation's physical configuration.
        Defaults to a reasonable default configuration.
    return_frames
        If True (default), return all color and velocity values as numpy arrays, with
        timesteps on axis 0. If False, return only the last color and velocity fields.
    disable_progress_bar
        If True, disable the printed progress bar.

    Returns
    -------
    color : jnp.ndarray
        If `return_frames` is False: the final color field, with shape [y, x, 3].
        Otherwise: each frame's color field, with shape [timesteps, y, x, 3].
    velocity : jnp.ndarray
        If `return_frames` is False: the final velocity field, with shape [y, x, 2].
        Otherwise: each frame's velocity field, with shape [timesteps, y, x, 2].
    """

    # Move `color` to the default device if it doesn't have one
    if isinstance(color, np.ndarray):
        color = jax.device_put(color)

    # Handle defaults
    if velocity is None:
        velocity = jnp.zeros((color.shape[0], color.shape[1], 2), color.dtype)
    if force is None:
        force = jnp.zeros((color.shape[0], color.shape[1], 2), color.dtype)

    # Initial pressure estimate: zero everywhere
    pressure = jnp.zeros((color.shape[0], color.shape[1]), color.dtype)

    # Pre-commit all arrays to the same device as `color`
    device = color.device_buffer.device()
    velocity = jax.device_put(velocity, device)
    force = jax.device_put(force, device)
    pressure = jax.device_put(pressure, device)

    if return_frames:
        color_frames = np.empty((timesteps, *color.shape), color.dtype)
        velocity_frames = np.empty((timesteps, *velocity.shape), velocity.dtype)

    for t in tqdm.trange(
        timesteps, disable=disable_progress_bar, desc="Simulating", unit="frame"
    ):
        color, velocity, pressure = step(color, velocity, force, pressure, config)

        if return_frames:
            color_frames[t] = color
            velocity_frames[t] = velocity

    if return_frames:
        return color_frames, velocity_frames

    return color, velocity
