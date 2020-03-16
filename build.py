"""Build script for project installation."""
import platform

from typing import Dict

from setuptools import Extension

ext_modules = [
    Extension(
        "signal_processing_algorithms.e_divisive.calculators._e_divisive",
        sources=["./src/signal_processing_algorithms/e_divisive/calculators/e_divisive.c"],
        extra_compile_args=["-O3"],
        extra_link_args=[] if "Darwin" in platform.system() else ["-shared"],
        optional=True,
    )
]


def build(setup_kwargs: Dict) -> None:
    """Build the extensions."""
    setup_kwargs.update({"ext_modules": ext_modules})
