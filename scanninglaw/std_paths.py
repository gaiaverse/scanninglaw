#!/usr/bin/env python
#
# std_paths.py
# Defines a set of paths used by scripts in the selectionfunctions module.
#
# Copyright (C) 2020 Douglas Boubert
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from __future__ import print_function, division

import os
from .config import config


script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir_default = os.path.abspath(os.path.join(script_dir, 'data'))
test_dir = os.path.abspath(os.path.join(script_dir, 'tests'))
output_dir_default = os.path.abspath(os.path.join(script_dir, 'output'))


def fix_path(path):
    """
    Returns an absolute path, with '~' expanded to the user's home directory.
    """
    return os.path.abspath(os.path.expanduser(path))


def data_dir():
    """
    Returns the directory used to store large data files (e.g., dust maps).
    """
    dirname = config.get('data_dir', data_dir_default)
    return fix_path(dirname)


def output_dir():
    """
    Returns a directory that can be used to store temporary output.
    """
    dirname = config.get('output_dir', output_dir_default)
    return fix_path(dirname)
