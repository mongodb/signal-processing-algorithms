from distutils import build_ext
from sys import platform

from setuptools import Extension
from setuptools.errors import DistutilsPlatformError

ext_modules=[
    Extension(
        "signal_processing_algorithms.e_divisive.calculators._e_divisive",
        sources=["./src/signal_processing_algorithms/e_divisive/calculators/e_divisive.c"],
        extra_compile_args=["-O3"],
        extra_link_args=[] if "Darwin" in platform.system() else ["-shared"],
        optional=True,
    )
]

class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )