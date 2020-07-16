#!/usr/bin/env python
#
# sfexceptions.py
# Defines exceptions for the selectionfunctions package.
#
# Copyright (C) 2020  Douglas Boubert
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

from . import std_paths

class Error(Exception):
    pass

class CoordFrameError(Error):
    pass


def data_missing_message(package, name):
    return ("The {name} selection function is not in the data directory:\n\n"
            "    {data_dir}\n\n"
            "To change the data directory, call:\n\n"
            "    from selectionfunctions.config import config\n"
            "    config['data_dir'] = '/path/to/data/directory'\n\n"
            "To download the {name} selection function to the data directory, call:\n\n"
            "    import selectionfunctions.{package}\n"
            "    selectionfunctions.{package}.fetch()\n").format(
                data_dir=std_paths.data_dir(),
                package=package,
                name=name)
