#!/usr/bin/env python3


# @Author: George Onoufriou <archer>
# @Date:   2018-09-05
# @Filename: setup.py
# @Last modified by:   archer
# @Last modified time: 2019-08-17
# @License: Please see LICENSE file in project root

import re
import subprocess
from setuptools import setup, find_packages, find_namespace_packages


def get_gitVersion():
    """Get the version from git describe in archlinux format."""
    try:
        # getting version from git as this is vcs
        # below equivelant or achlinux versioning scheme:
        # git describe --long | sed 's/\([^-]*-\)g/r\1/;s/-/./g
        git_describe = subprocess.Popen(
            ["git", "describe", "--long"],
            stdout=subprocess.PIPE)
        version_num = subprocess.check_output(
            ["sed", r"s/\([^-]*-\)g/r\1/;s/-/./g"],
            stdin=git_describe.stdout)
        git_describe.wait()
        version_git = version_num.decode("ascii").strip()

    except subprocess.CalledProcessError:
        # for those who do not have git or sed availiable (probably non-linux)
        # this is tricky to handle, lots of suggestions exist but none that
        # neither require additional library or subprocessess
        version_git = "0.0.1"  # for now we will provide a number for you
    return version_git


def get_requirements(path=None):
    """get a list of requirements and any dependency links associated.

    This function fascilitates git urls being in requirements.txt
    and installing them as normall just like pip install -r requirements.txt
    would but setup does not by default.

    :e.g requirements.txt\::

        ConfigArgParse
        git+https://github.com/DreamingRaven/python-ezdb.git#branch=master

    should result in

    :requirements\::

        [
            ConfigArgParse
            python-ezdb
        ]

    : dependenc links\::

        ["git+https://github.com/DreamingRaven/python-ezdb.git#egg=python-ezdb"]

    version can also be appended but that may break compatibility with
    pip install -r requirements.txt so we will not attempt that here but would
    look something like this:

    "git+https://github.com/DreamingRaven/python-ezdb.git#egg=python-ezdb-0.0.1"
    """
    dependency_links = []

    #  read in the requirements file
    path = path if path is not None else "./requirements.txt"
    with open(path, "r") as f:
        requirements = f.read().splitlines()

    # apply regex to find all desired groups like package name and url
    re_git_url = r"^\bgit.+/(.+)\.git"
    re_groups = list(map(lambda x: re.search(re_git_url, x), requirements))

    # iterate over regex and select package name group to insert over url
    for i, content in enumerate(re_groups):
        # re.search can return None so only if it returned something
        if(content):
            print(i, content, requirements[i])
            requirements[i] = content.group(1)
            dependency_links.append("{}#egg={}".format(content.group(0),
                                                       content.group(1)))

    return requirements, dependency_links


# get sourcecode version usually from git with some fallbacks
version = get_gitVersion()
print("version:", version)

# get dependencys and dependency install links to work with git urls
dependencies, links = get_requirements()
print("requirements:", dependencies)
print("requirement_links:", links)

# read in readme as readme for package as im too lazy to write a new one for
# package
with open("README.rst", "r") as fh:
    readme = fh.read()

# collect namespace packages but ignore certain undesired directories
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "examples", "examples.*", "tests", "tests.*",
             "build", "build.*", "dist", "dist.*", "venv", "venv.*"))
print("namespace packages:", packages)

setup(
    name="reseal",
    version=version,
    description="Template repository.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="George Onoufriou",
    url="https://github.com/DreamingRaven/python-reseal",
    packages=find_namespace_packages(),
    # scripts=['pyrtd'],
    install_requires=dependencies,
    dependency_links=links,
)
