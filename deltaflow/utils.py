"""
Various handy visualization utilities.

All images are expected to have values from 0 to 1.
"""

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from typing import Optional


def _get_velocity_quiver_uvc(velocity: np.ndarray, num_arrows: int = 15):
    resolution_x, resolution_y = velocity.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.mgrid[num_arrows : resolution_x : resolution_x // num_arrows],
        np.mgrid[num_arrows : resolution_y : resolution_y // num_arrows],
    )
    arrow_coords = velocity[y_coords, x_coords]
    return x_coords, y_coords, arrow_coords[:, :, 1], -arrow_coords[:, :, 0]


def draw_frame(color: np.ndarray, velocity: Optional[np.ndarray] = None, axes=None):
    """
    Draw a single frame of simulation.

    Parameters
    ----------
    color
        A color field. *Shape: [y, x, 3]*.
    velocity
        A velocity field. If provided, draws and returns a quiver plot; otherwise,
        just draws and returns an image. *Shape: [y, x, 2]*.
    axes
        If provided, uses these pyplot axes to draw the plots.
        Otherwise, create a new figure.
    """

    if axes is None:
        fig, axes = plt.subplots()
        axes.axis("off")

    img = axes.imshow(np.clip(color, 0.0, 1.0), vmin=0, vmax=1)

    if velocity is None:
        return img

    quiver = axes.quiver(*_get_velocity_quiver_uvc(velocity), color="white", width=0.01)
    return img, quiver


def animate_frames(
    color_frames: np.ndarray,
    velocity_frames: Optional[np.ndarray] = None,
    interval: int = 30,
) -> animation.FuncAnimation:
    """
    Draw a stack of frames into a matplotlib animation.

    Parameters
    ----------
    color
        A stack of color field frames. *Shape: [timesteps, y, x, 3]*.
    velocity
        A stack of velocity field frames. If provided, draws a quiver plot on each
        frame. *Shape: [timesteps, y, x, 2]*.
    interval
        Milliseconds between consecutive frames of the animation.
    """

    fig, ax = plt.subplots()
    plt.close(fig)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis("off")

    if velocity_frames is None:
        img = draw_frame(color_frames[0], axes=ax)
    else:
        img, quiver = draw_frame(color_frames[0], velocity_frames[0], ax)

    def set_data(i):
        img.set_data(np.clip(color_frames[i], 0.0, 1.0))

        if velocity_frames is None:
            return (img,)

        _, _, arrow_x, arrow_y = _get_velocity_quiver_uvc(velocity_frames[i])
        quiver.set_UVC(arrow_x, arrow_y)
        return img, quiver

    return animation.FuncAnimation(
        fig,
        set_data,
        interval=interval,
        frames=color_frames.shape[0],
        blit=True,
        repeat=False,
    )


### Video writer
# Code modified from kylemcdonald's python-utils (https://github.com/kylemcdonald/python-utils)
# which is released which is under the MIT license. Copyright notice reproduced below:

# Copyright (c) 2018 Kyle McDonald
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.


class _VideoWriter:
    def __init__(
        self,
        fn,
        vcodec="libx264",
        fps=60,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.fn = fn
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = vcodec

    def add(self, frame):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(w, h),
                    **self.input_args
                )
                .output(self.fn, **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def write_video(filepath: str, images: np.ndarray, **kwargs) -> None:
    """
    Write a stack of color frames into a video.

    Parameters
    ----------
    filepath
        The filepath to save the video to.
    images
        A stack of color frames. *Shape: [timesteps, y, x, 3].*
    kwargs
        See `deltaflow.utils._VideoWriter` for all optional arguments.
    """
    writer = _VideoWriter(filepath, **kwargs)
    for image in images * 255:
        writer.add(image)
    writer.close()
