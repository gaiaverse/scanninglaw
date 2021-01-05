#!/usr/bin/env python
#
# times.py
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
import tqdm
import scipy

import astropy.coordinates as coordinates
import astropy.units as units
import pandas as pd
import healpy as hp
from scipy import interpolate, special, spatial

from .std_paths import *
from .map import ScanningLaw, ensure_flat_icrs, coord2healpix, tcbgaia2obmt, obmt2tcbgaia
from .source import ensure_gaia_g, Source
from . import fetch_utils

from time import time


class asf(ScanningLaw):
    """
    Queries the Gaia DR2 selection function (Boubert & Everall, 2019).
    """

    def __init__(self, map_fname=None, version='cogiv_2020', sample='Astrometry'):
        """
        Args:
            map_fname (Optional[:obj:`str`]): Filename of the Boubert,Everall,Holl 2020 scanning law. Defaults to
                :obj:`None`, meaning that the default location is used.
            version (Optional[:obj:`str`]): The scanning law version to download. Valid versions
                are :obj:`'cogiv_2020'`
                Defaults to :obj:`'cogiv_2020'`.
            sample (Optional[:obj:`str`]): The Gaia data sample being used. Valid options are
                are :obj:`'Astrometry'`
                are :obj:`'Photometry'`
                are :obj:`'Spectroscopy'`
                Defaults to :obj:`'Astrometry'`.
        """

        if version=='cog': version='cogiv_2020'
        if map_fname is None:
            map_fname = os.path.join(data_dir(), 'cog', '{}'.format('cog_dr2_asf_v1.h5'))

        self.asf_fname = os.path.join(data_dir(), 'cog', map_fname)

        t_start = time()

        # Load auxilliary data
        print('Loading auxilliary data ...')
        _box={}; keys=['magbin', 'varal_50', 'varal_16', 'varal_84', 'r_50', 'good_frac_50']
        with h5py.File(self.asf_fname, 'r') as hf:
            for key in keys:
                _box[key]=hf[key][...]
            self._nside = hf['nside'][...]
            self.matrix_map=hf['matrix_map'][...]
            #self.D_array = hf['D'][...]

        # R AC already implicitly included in varal_50 which is from <ngood/(P_aa + P_dd)>
        self.rho_interp = scipy.interpolate.interp1d(_box['magbin']+0.05, _box['good_frac_50']/_box['varal_50'])
        #self.sigAL_interp = scipy.interpolate.interp1d(_box['magbin']+0.05, np.sqrt(_box['varal_50'] * (1+_box['r_50'] * (92/520)**2)))

        self.sp_bins = np.array([5, 13,  16, 16.3, 17, 17.2, 18, 18.1, 19, 19.05, 19.95,
                                 20, 20.3, 20.4, 20.5, 20.6, 20.7,20.8,20.9, 21])

        t_auxilliary = time()

        t_finish = time()

        print('t = {:.3f} s'.format(t_finish - t_start))
        print('  auxilliary: {: >7.3f} s'.format(t_auxilliary-t_start))

    def _get_spidx(self, G):

        """
        Returns the magnitude bin ids for the given magnitudes.

        Args:
            G (:obj:`np.ndarray`): G magnitude.

        Returns:
            magidx (:obj:`dict`): bin ID of magnitude

        """

        try: magidx = np.zeros(G.shape).astype(int) - 1
        except AttributeError:  magidx = np.zeros(np.array([G]).shape).astype(int) - 1

        for mag in self.sp_bins:
            magidx += (G>mag).astype(int)

        magidx[magidx==len(self.sp_bins)-1] = -99
        magidx[magidx==-1] = -99

        return magidx

    #@ensure_flat_icrs
    @ensure_gaia_g
    def query(self, sources):
        """
        Returns the scanning law at the requested coordinates.

        Args:
            sources (:obj:`astropy.coordinates.SkyCoord`): The coordinates to query.
                    (:obj:`scanninglaw.source.Source`): The coordinates to query.

        KwArgs:
            return_counts
            return_fractions
            fov

        Returns:
            (:obj:`dict`): Observation times of each object by each FoV. Number of observations of each object by each FoV

        """

        # Convert coordinates to healpix indices
        hpxidx = coord2healpix(sources.coord, 'icrs', self._nside, nest=True)
        if type(hpxidx)==np.int64:
            singular=True
            hpxidx = np.array([hpxidx])
        else: singular=False

        # Extract Gaia G magnitude
        G = sources.photometry.measurement['gaia_g']
        sp_idx = self._get_spidx(G)

        D = np.zeros(hpxidx.shape+(15,))
        with h5py.File(self.asf_fname, 'r') as hf:
            for ii in range(len(self.sp_bins)-1):
                D[sp_idx==ii]=hf['D'][str(ii)][hpxidx[sp_idx==ii]]

        #D = self.D_array[hpxidx,:,sp_idx]
        precision = np.zeros(hpxidx.shape+(5,5))
        for i in range(self.matrix_map.shape[0]):
            precision[...,self.matrix_map[i,1], self.matrix_map[i,2]] = \
                    D[...,self.matrix_map[i,0]]*(8+6./7)

        covariance = np.moveaxis(np.linalg.inv(precision),[-2,-1],[0,1])
        rho = self.rho_interp(G)

        if singular: return covariance[:,:,0]/rho
        return covariance/rho


def fetch(version='cogiv_2020', fname=None):
    """
    Downloads the specified version of the Gaia DR2 scanning law.

    Args:
        version (Optional[:obj:`str`]): The map version to download. Valid versions are
            :obj:`'cogi_2020'` (Boubert, Everall & Holl 2020)

    Raises:
        :obj:`ValueError`: The requested version of the map does not exist.

        :obj:`DownloadError`: Either no matching file was found under the given DOI, or
            the MD5 sum of the file was not as expected.

        :obj:`requests.exceptions.HTTPError`: The given DOI does not exist, or there
            was a problem connecting to the Dataverse.
    """

    #local_fname = os.path.join(data_dir(), 'cog', '{}.csv.gz'.format(version))


    doi = {'cogiv_2020': '10.7910/DVN/FURYBN'}
    # Raise an error if the specified version of the map does not exist
    try:
        doi = doi[version]
    except KeyError as err:
        raise ValueError('Version "{}" does not exist. Valid versions are: {}'.format(
            version,
            ', '.join(['"{}"'.format(k) for k in doi.keys()])
        ))

    requirements = {
        'cogiv_2020': {'filename': 'cog_dr2_asf_v1.h5'},
    }[version]
    local_fname = os.path.join(data_dir(), 'cog', requirements['filename'])

    # Download the data
    fetch_utils.dataverse_download_doi(
        doi,
        local_fname,
        file_requirements=requirements)
