"""Package setup for signal_processing_algorithms."""
import platform
from src.signal_processing_algorithms import __package_name__, __version__
from setuptools import setup, find_packages, Extension


setup(
    name=__package_name__,
    version=__version__,
    description="Signal Processing Algorithms from MongoDB",
    python_requires=">=3",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "misc-utils-py~=0.1.2",
        "numpy~=1.16",
        "scipy~=1.3",
        "structlog~=19.1"
    ],
    zip_safe=False,
    include_package_data=True,
    ext_modules=[
        Extension(
            "signal_processing_algorithms.e_divisive.calculators._e_divisive",
            sources=["./src/signal_processing_algorithms/e_divisive/calculators/e_divisive.c"],
            extra_compile_args=["-O3"],
            extra_link_args=[] if "Darwin" in platform.system() else ["-shared"],
            optional=True,
        )
    ],
)
