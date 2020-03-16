import subprocess

import pkg_resources

import signal_processing_algorithms


class TestPackageVersion(object):
    def test_version_is_updated(self):
        package = pkg_resources.get_distribution(signal_processing_algorithms.__name__)
        pip_command = "pip install {package_name}".format(package_name=package.project_name)
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
        assert package.version not in versions
