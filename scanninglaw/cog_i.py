#!/usr/bin/env python
#
# cog_ii.py
# Reads the Gaia DR2 selection function from Completeness
# of the Gaia-verse Paper II, Boubert & Everall (2020).
#
# Copyright (C) 2020  Douglas Boubert & Andrew Everall
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
import h5py
import numpy as np

import astropy.coordinates as coordinates
import astropy.units as units
import h5py
import healpy as hp
from scipy import interpolate, special

from .std_paths import *
from .map import ScanningLaw, ensure_flat_icrs, coord2healpix
from .source import ensure_gaia_g
from . import fetch_utils

from time import time


class dr2_sf(ScanningLaw):
    """
    Queries the Gaia DR2 selection function (Boubert & Everall, 2019).
    """

    def __init__(self, map_fname=None, version='modelAB', crowding=False, bounds=True):
        """
        Args:
            map_fname (Optional[:obj:`str`]): Filename of the BoubertEverall2019 selection function. Defaults to
                :obj:`None`, meaning that the default location is used.
            version (Optional[:obj:`str`]): The selection function version to download. Valid versions
                are :obj:`'modelT'` and :obj:`'modelAB'`
                Defaults to :obj:`'modelT'`.
            crowding (Optional[:obj:`bool`]): Whether or not the selection function includes crowding.
                Defaults to :obj:`'False'`.
            bounds (Optional[:obj:`bool`]): Whether or not the selection function is bounded to 0.0 < G < 25.0.
                Defaults to :obj:`'True'`.
        """

        if map_fname is None:
            map_fname = os.path.join(data_dir(), 'cog_ii', 'cog_ii_dr2.h5')

        t_start = time()

        with h5py.File(map_fname, 'r') as f:
            # Load auxilliary data
            print('Loading auxilliary data ...')


            t_auxilliary = time()


            t_sf = time()


        t_interpolator = time()

        t_finish = time()

        print('t = {:.3f} s'.format(t_finish - t_start))
        print('  auxilliary: {: >7.3f} s'.format(t_auxilliary-t_start))
        print('          sf: {: >7.3f} s'.format(t_sf-t_auxilliary))
        print('interpolator: {: >7.3f} s'.format(t_interpolator-t_sf))

    def _scanning_law(self,_n,_parameters):



        return _result


    @ensure_flat_icrs
    @ensure_gaia_g
    def query(self, sources):
        """
        Returns the selection function at the requested coordinates.

        Args:
            coords (:obj:`astropy.coordinates.SkyCoord`): The coordinates to query.

        Returns:
            Selection function at the specified coordinates, as a fraction.

        """

        # Convert coordinates to healpix indices
        hpxidx = coord2healpix(sources.coord, 'icrs', self._nside, nest=True)

        # Calculate the number of observations of each source
        n = self._n_field[hpxidx]

        # Extract Gaia G magnitude
        G = sources.photometry.measurement['gaia_g']

        if self._crowding == True:

            # Work out HEALPix index in crowding nside
            hpxidx_crowding = np.floor(hpxidx * hp.nside2npix(self._nside_crowding) / hp.nside2npix(self._nside)).astype(np.int)

            # Calculate the local density field at each source
            log10_rho = self._log10_rho_field[hpxidx_crowding]

            # Calculate parameters
            sf_parameters = self._interpolator(log10_rho,G)

        else:

            # Calculate parameters
            sf_parameters = self._interpolator(G)

        # Evaluate selection function
        selection_function = self._selection_function(n,sf_parameters)

        if self._bounds == True:
            _outside_bounds = np.where( (G<self._g_min) | (G>self._g_max) )
            selection_function[_outside_bounds] = 0.0

        return selection_function


def fetch():
    """
    Downloads the specified version of the Bayestar dust map.

    Args:
        version (Optional[:obj:`str`]): The map version to download. Valid versions are
            :obj:`'bayestar2019'` (Green, Schlafly, Finkbeiner et al. 2019),
            :obj:`'bayestar2017'` (Green, Schlafly, Finkbeiner et al. 2018) and
            :obj:`'bayestar2015'` (Green, Schlafly, Finkbeiner et al. 2015). Defaults
            to :obj:`'bayestar2019'`.

    Raises:
        :obj:`ValueError`: The requested version of the map does not exist.

        :obj:`DownloadError`: Either no matching file was found under the given DOI, or
            the MD5 sum of the file was not as expected.

        :obj:`requests.exceptions.HTTPError`: The given DOI does not exist, or there
            was a problem connecting to the Dataverse.
    """

    doi = '10.7910/DVN/OFRA78'

    requirements = {'filename': 'cog_dr2_scanning_law_v1.csv.gz'}

    local_fname = os.path.join(data_dir(), 'cog_i', 'cog_dr2_scanning_law_v1.csv.gz')

    # Download the data
    fetch_utils.dataverse_download_doi(
        doi,
        local_fname,
        file_requirements=requirements)
