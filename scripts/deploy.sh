#!/usr/bin/env bash

rm -rf dist

if [ -z "$1" ]; then
    echo "Missing PyPi username"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Missing PyPi password"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Missing PyPi repository url"
    exit 1
fi

username=$1
password=$2
repo=$3

python build.py sdist bdist_wheel
rc=$?

if [ $rc -ne 0 ]; then
    echo "Error building distribution"
    exit 3
fi

twine upload --username $username --password $password --repository-url $repo dist/*

git tag $(python build.py --version)
git push --tags

exit $?
