import subprocess

from signal_processing_algorithms import __package_name__, __version__


class TestPackageVersion(object):
    def test_version_is_updated(self):
        pip_command = "pip install {package_name}".format(package_name=__package_name__)
        get_versions_command = (
            pip_command
            + r"==invalidversion 2>&1 \
                                | grep -oE '(\(.*\))' \
                                | awk -F:\  '{print$NF}' \
                                | sed -E 's/( |\))//g' \
                                | tr ',' '\n'"
        )
        versions = subprocess.check_output(get_versions_command, shell=True).decode("UTF-8")
        versions = [x for x in versions.split("\n") if len(x) > 0]
        assert __version__ not in versions
