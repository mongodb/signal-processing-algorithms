import sys
import warnings
import distutils.core
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from src.signal_processing_algorithms import __package_name__, __version__

import platform
from setuptools import setup, find_packages


class CustomBuildExt(build_ext):
    """
    Allow C extension building to fail.

    The C extension speeds up E-Divisive calculation, but is not essential.
    """

    warning_message = """
********************************************************************
WARNING: %s could not
be compiled. No C extensions are essential for signal processing to run,
although they do result in significant speed improvements.
%s
"""

    def run(self):
        """
        Run a custom build, errors are ignored.
        """
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            e = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(e))
            warnings.warn(
                self.warning_message
                % (
                    "Extension modules",
                    "There was an issue with " "your platform configuration" " - see above.",
                )
            )

    def build_extension(self, ext):
        """
        Build the extension, ignore any errors.
        """
        name = ext.name
        try:
            build_ext.build_extension(self, ext)
        except build_errors:
            e = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(e))
            warnings.warn(
                self.warning_message
                % ("The %s extension " "module" % (name,), "failed to compile.")
            )


if sys.platform == "win32":
    # distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    build_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError)
else:
    build_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

ext_modules = [
    distutils.core.Extension(
        "signal_processing_algorithms/_e_divisive",
        sources=["./src/signal_processing_algorithms/e_divisive.c"],
        extra_compile_args=["-O3"],
        extra_link_args=[] if "Darwin" in platform.system() else ["-shared"],
    )
]
extra_opts = {}

if "--no_ext" in sys.argv:
    sys.argv.remove("--no_ext")
elif sys.platform.startswith("java") or sys.platform == "cli" or "PyPy" in sys.version:
    sys.stdout.write(
        """
*****************************************************\n
The optional C extensions are currently not supported\n
by this python implementation.\n
*****************************************************\n
"""
    )
else:
    extra_opts["ext_modules"] = ext_modules

setup(
    name=__package_name__,
    version=__version__,
    description="Algorithms from MongoDB",
    python_requires=">=3",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy~=1.16", "scipy~=1.3", "structlog~=19.1"],
    zip_safe=False,
    cmdclass={"build_ext": CustomBuildExt},
    include_package_data=True,
    # fmt: off
    # Trailing comma causes misleading error message on py2k.
    **extra_opts
    # fmt: on
)
