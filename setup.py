from distutils.core import setup

setup(
    name="DeltaFlow",
    version="1.0",
    description="a GPU-accelerated differentiable fluid simulator written in JAX",
    long_description=open("README.md").read(),
    url="https://github.com/maxwells-daemons/deltaflow",
    packages=["deltaflow"],
    license="MIT",
    author="Aidan Swope",
    author_email="aidanswope@gmail.com",
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib",
        "numpy",
        "tqdm",
        "pillow",
        "ffmpeg-python",
    ],
    extras_require={"docs": ["sphinx", "furo"]},
)
