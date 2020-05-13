#!/usr/bin/env bash
set -e
set -o pipefail

# This uses 'manylinux' docker image and 'auditwheel' to ensure our c extension
# works on lots of linux platforms without needing to be recompiled. The image
# provides toolchain for various version. We will be using the 3.7 version which
# is at the path below.
#
# For more information see: https://github.com/pypa/auditwheel
export PATH="$PATH:/opt/python/cp37-cp37m/bin"

poetry build
find ./dist -name "*.whl" | xargs auditwheel repair
rm dist/*.whl
mv wheelhouse/* dist
echo poetry publish --username $PYPI_USERNAME --password $PYPI_PASSWORD
