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


class dr2_asf(ScanningLaw):
    """
    Queries the Gaia DR2 selection function (Boubert & Everall, 2019).
    """

    def __init__(self, map_fname=None, version='cogi_2020', sample='Astrometry',
                        fractions='cog_dr2_gaps_and_fractions_v1.h5', require_persistent=False, test=False):
        """
        Args:
            map_fname (Optional[:obj:`str`]): Filename of the Boubert,Everall,Holl 2020 scanning law. Defaults to
                :obj:`None`, meaning that the default location is used.
            version (Optional[:obj:`str`]): The scanning law version to download. Valid versions
                are :obj:`'cogi_2020'`
                are :obj:`'dr2_nominal'`
                Defaults to :obj:`'cogi_2020'`.
            sample (Optional[:obj:`str`]): The Gaia data sample being used. Valid options are
                are :obj:`'Astrometry'`
                are :obj:`'Photometry'`
                Defaults to :obj:`'Spectroscopy'`.
        """

        if version=='cog': version='cogi_2020'
        if map_fname is None:
            map_fname = os.path.join(data_dir(), 'cog', '{}.csv'.format(version))
        gaps_fname = os.path.join(data_dir(), 'cog', '{}.csv'.format(sample))
        fractions_fname = os.path.join(data_dir(), 'cog', fractions)

        sigma_al_fname = '/data/asfe2/Projects/gaia_psf/sigmaAL_5d_mag_med.h'
        self.cov_filename = '/data/asfe2/Projects/gaia_psf/scanninglaw_prec5d_2015p5T_bisect_128_maggaps_fracweights.h'

        t_start = time()

        # Load auxilliary data
        print('Loading auxilliary data ...')
        _box = {}
        with h5py.File(sigma_al_fname, 'r') as hf:
            for key in hf.keys():
                _box[key] = hf[key][...]
        self.sigAL_interp = scipy.interpolate.interp1d(_box['magbin'], _box['sigmaal_50'] * (1+_box['r_50'] * (92/520)**2))

        self._nside = 128
        self.magbins = np.array([5, 13,  16, 16.3, 17, 17.2, 18, 18.1, 19, 19.05, 19.95,
                                 20, 20.3, 20.4, 20.5, 20.6, 20.7,20.8,20.9, 21])

        t_auxilliary = time()

        t_finish = time()

        print('t = {:.3f} s'.format(t_finish - t_start))
        print('  auxilliary: {: >7.3f} s'.format(t_auxilliary-t_start))

    def _get_magidx(self, G):

        """
        Returns the magnitude bin ids for the given magnitudes.

        Args:
            G (:obj:`np.ndarray`): G magnitude.

        Returns:
            magidx (:obj:`dict`): bin ID of magnitude

        """

        try: magidx = np.zeros(G.shape).astype(int) - 1
        except AttributeError:  magidx = np.zeros(np.array([G]).shape).astype(int) - 1

        for mag in self.magbins:
            magidx += (G>mag).astype(int)

        magidx[magidx==len(self.magbins)-1] = -99
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
        G_idx = self._get_magidx(G)

        _box = {};  keys=['D'];
        D = np.zeros((len(G_idx), 15))
        magidxs = np.arange(len(self.magbins)-1).astype(str)
        for ii in range(len(magidxs)):
            with h5py.File(self.cov_filename, 'r') as hf:
                matrix_map=hf['matrix_map'][...]
                for key in keys:
                    D[G_idx==ii] = hf[magidxs[ii]][key][hpxidx[G_idx==ii]]

        precision = np.zeros((5,5,D[:,0].shape[0]))
        for i in range(matrix_map.shape[0]):
            precision[matrix_map[i,1], matrix_map[i,2]] = \
                D[:,matrix_map[i,0]]*(8+6./7)

        covariance = np.moveaxis(np.linalg.inv(np.moveaxis(precision,2,0)),0,2)
        sigma_al = self.sigAL_interp(G)

        if singular: return covariance[:,:,0]*sigma_al**2
        return covariance*sigma_al**2


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

    local_fname = os.path.join(data_dir(), 'cog', '{}.csv.gz'.format(version))

    if (version=='dr2_nominal')&(fname is None):
        raise ValueError("\nNominal scanning law at ftp.cosmos.esa.int/GAIA_PUBLIC_DATA/GaiaScanningLaw/DEOPTSK-1327_Gaia_scanlaw.csv.gz.\n"\
                          "Download .gz file using ftp client then run fetch(version='dr2_nominal', fname='path/to/file'). ")
    elif version=='dr2_nominal':
        fetch_utils.move_file_location(fname, local_fname)
        return None


    doi = {
        'cogi_2020': '10.7910/DVN/OFRA78',
        'dr2_nominal': ''
    }
    # Raise an error if the specified version of the map does not exist
    try:
        doi = doi[version]
    except KeyError as err:
        raise ValueError('Version "{}" does not exist. Valid versions are: {}'.format(
            version,
            ', '.join(['"{}"'.format(k) for k in doi.keys()])
        ))

    requirements = {
        'cogi_2020': {'filename': 'cog_dr2_scanning_law_v1.csv.gz'},
    }[version]

    # Download the data
    fetch_utils.dataverse_download_doi(
        doi,
        local_fname,
        file_requirements=requirements)
