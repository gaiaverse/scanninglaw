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
import tqdm

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


class dr2_sl(ScanningLaw):
    """
    Queries the Gaia DR2 selection function (Boubert & Everall, 2019).
    """

    def __init__(self, map_fname=None, version='cogi_2020', sample='Astrometry', require_persistent=False, test=False):
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

        if map_fname is None:
            map_fname = os.path.join(data_dir(), 'cog', '{}.csv'.format(version))

        gaps_fname = os.path.join(data_dir(), 'cog', '{}.csv'.format(sample))

        t_start = time()

        # Load auxilliary data
        print('Loading auxilliary data ...')
        ##### Load in scanning law
        _columns = ['JulianDayNumberRefEpoch2010TCB@Gaia', 'JulianDayNumberRefEpoch2010TCB@Barycentre_1', 'JulianDayNumberRefEpoch2010TCB@Barycentre_2',
                    'ra_FOV_1(deg)', 'dec_FOV_1(deg)', 'scanPositionAngle_FOV_1(deg)', 'ra_FOV_2(deg)', 'dec_FOV_2(deg)', 'scanPositionAngle_FOV_2(deg)']
        if test: _data = pd.read_csv(map_fname, usecols=_columns, nrows=1000000)
        else: _data = pd.read_csv(map_fname, usecols=_columns)
        _keys = ['tcb_at_gaia','tcb_at_bary1','tcb_at_bary2','ra_fov_1','dec_fov_1','angle_fov_1','ra_fov_2','dec_fov_2','angle_fov_2']
        _box = {}
        for j,k in zip(_columns,_keys): _box[k] = _data[j].values
        _box['scan_idx'] = np.arange(len(_box['ra_fov_1']))

        ##### Load gaps
        _columns = ['start [rev]', 'end [rev]', 'persistent'];
        _data = pd.read_csv(gaps_fname, usecols=_columns)
        self._gaps = np.vstack((_data['start [rev]'].values, _data['end [rev]'].values)).T
        if require_persistent: self._gaps=self._gaps[_data['persistent']==True]
        if sample=='Astrometry':
            self._gaps = np.vstack((np.array([-np.inf, obmt2tcbgaia(1192.13)])[np.newaxis,:], self._gaps ))
            self._gaps = np.vstack(( self._gaps, np.array([obmt2tcbgaia(3750.56), np.inf])[np.newaxis,:],  ))

        t_auxilliary = time()

        # Compute 3D spherical projections
        self.xyz_fov_1 = np.stack([np.cos(np.deg2rad(_box['ra_fov_1']))*np.cos(np.deg2rad(_box['dec_fov_1'])),
                              np.sin(np.deg2rad(_box['ra_fov_1']))*np.cos(np.deg2rad(_box['dec_fov_1'])),
                              np.sin(np.deg2rad(_box['dec_fov_1']))]).T
        self.xyz_fov_2 = np.stack([np.cos(np.deg2rad(_box['ra_fov_2']))*np.cos(np.deg2rad(_box['dec_fov_2'])),
                              np.sin(np.deg2rad(_box['ra_fov_2']))*np.cos(np.deg2rad(_box['dec_fov_2'])),
                              np.sin(np.deg2rad(_box['dec_fov_2']))]).T
        ##### Compute rotation matrices
        _xaxis = self.xyz_fov_1
        _zaxis = np.cross(self.xyz_fov_1,self.xyz_fov_2)
        _yaxis = -np.cross(_xaxis,_zaxis)
        _yaxis /= np.linalg.norm(_yaxis,axis=1)[:,np.newaxis]
        _zaxis /= np.linalg.norm(_zaxis,axis=1)[:,np.newaxis]
        _uaxis = np.array([1,0,0])
        _vaxis = np.array([0,1,0])
        _waxis = np.array([0,0,1])
        self._matrix = np.moveaxis(np.stack((_xaxis, _yaxis, _zaxis)), 1,0)

        self.tcb_at_gaia = _box['tcb_at_gaia'].copy()

        t_sf = time()

        # Gaia FoV parameters
        self.t_diff = 1/24 # 1 hours
        self.r_search = np.tan(np.deg2rad(0.35*np.sqrt(2)))
        b_fov = 0.35
        zeta_origin_1 = +221/3600
        zeta_origin_2 = -221/3600
        self.b_upp_1 = b_fov+zeta_origin_1
        self.b_low_1 = -b_fov+zeta_origin_1
        self.b_upp_2 = b_fov+zeta_origin_2
        self.b_low_2 = -b_fov+zeta_origin_2
        self.l_fov_1 = 0.0
        self.l_fov_2 = 106.5

        self.tree_fov1 = spatial.cKDTree(self.xyz_fov_1)
        self.tree_fov2 = spatial.cKDTree(self.xyz_fov_2)

        t_interpolator = time()

        t_finish = time()

        print('t = {:.3f} s'.format(t_finish - t_start))
        print('  auxilliary: {: >7.3f} s'.format(t_auxilliary-t_start))
        print('          sf: {: >7.3f} s'.format(t_sf-t_auxilliary))
        print('interpolator: {: >7.3f} s'.format(t_interpolator-t_sf))

    def linearbisect(self, xyz_obj, xyz_line, tgaia_line):

        _d_line = xyz_line[1]-xyz_line[0]
        _d_obj = xyz_obj-xyz_line[0]
        _line_sqlen = np.sum((_d_line)**2, axis=1)

        _d_tobs = (tgaia_line[1]-tgaia_line[0]) * np.sum(_d_line * _d_obj, axis=1) / _line_sqlen

        return tgaia_line[0]+_d_tobs

    def _scanning_law(self, xyz_source):

        # Run this code for large numbers of sources (5 million +)

        nsource = xyz_source.shape[0]
        t_previous = -99999*np.ones(nsource)
        t_box = {i:[] for i in range(nsource)}
        idx_box = {i:[] for i in range(nsource)}

        tgaia_fov1 = [[] for i in range(nsource)]
        tgaia_fov2 = [[] for i in range(nsource)]
        nscan_fov1 = [0 for i in range(nsource)]
        nscan_fov2 = [0 for i in range(nsource)]

        tree_source = spatial.cKDTree(xyz_source)

        # Iterate through scanning time steps
        for _tidx in tqdm.tqdm_notebook(range(0,self.xyz_fov_1.shape[0])):
            _t_now = self.tcb_at_gaia[_tidx]
            # Find all sources in scan window
            _in_fov = tree_source.query_ball_point([self.xyz_fov_1[_tidx].copy(order='C'),
                                                    self.xyz_fov_2[_tidx].copy(order='C')],self.r_search)
            n_fov1 = len(_in_fov[0])
            _in_fov = _in_fov[0]+_in_fov[1]
            if len(_in_fov) == 0:
                continue

            # xyz coordinates of sources
            _xyz = xyz_source[_in_fov]
            _uvw = np.einsum('ij,nj->ni',self._matrix[_tidx],_xyz)

            _l = np.rad2deg(np.arctan2(_uvw[:,1],_uvw[:,0]))
            _b = np.rad2deg(np.arctan2(_uvw[:,2],np.sqrt(_uvw[:,0]**2.0+_uvw[:,1]**2.0)))

            condition = (((np.abs(_l-self.l_fov_1)<1.0)&(_b<self.b_upp_1)&(_b>self.b_low_1))\
                               |((np.abs(_l-self.l_fov_2)<1.0)&(_b<self.b_upp_2)&(_b>self.b_low_2)))\
                              &(_t_now-t_previous[_in_fov]>self.t_diff)
            _where = np.where(condition)[0]
            _valid = np.array(_in_fov,dtype=np.int)[_where]
            n_fov1 = np.sum(condition[:n_fov1]);

            #if len(_valid)>0:
            if np.sum(condition)>0:
                tcbgaia_fov1 = self.linearbisect(_xyz[_where[:n_fov1]], self.xyz_fov_1[max(0,_tidx-1):_tidx+2][:2][:,np.newaxis],
                                                                        self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2][:,np.newaxis])
                tcbgaia_fov2 = self.linearbisect(_xyz[_where[n_fov1:]], self.xyz_fov_2[max(0,_tidx-1):_tidx+2][:2][:,np.newaxis],
                                                                        self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2][:,np.newaxis])

                tcbgaia = np.hstack((tcbgaia_fov1, tcbgaia_fov2))
                condition_gap = np.ones(len(tcbgaia)).astype(bool)
                for ii in range(self._gaps.shape[0]):
                    condition_gap = condition_gap&( (tcbgaia<self._gaps[ii,0])|(tcbgaia>self._gaps[ii,1]) )

                _valid = _valid[condition_gap]
                n_fov1 = np.sum(condition_gap[:n_fov1]);

                t_previous[_valid] = _t_now

                for ii in range(n_fov1):
                    _sidx = _valid[ii]
                    tgaia_fov1[_sidx].append(tcbgaia_fov1[ii])
                    nscan_fov1[_sidx] += 1
                    #print(self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2])
                for ii in range(len(_valid)-n_fov1):
                    _sidx = _valid[ii+n_fov1]
                    tgaia_fov2[_sidx].append(tcbgaia_fov2[ii])
                    nscan_fov2[_sidx] += 1
                    #print(self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2])

        return tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2

    def _scanning_law_inverse(self, xyz_source):

        # Run this code for small numbers of sources (less than 5 million)

        nsource = xyz_source.shape[0]
        t_previous = -99999*np.ones(nsource)
        t_box = {i:[] for i in range(nsource)}
        idx_box = {i:[] for i in range(nsource)}

        tgaia_fov1 = [[] for i in range(nsource)]
        tgaia_fov2 = [[] for i in range(nsource)]
        nscan_fov1 = [0 for i in range(nsource)]
        nscan_fov2 = [0 for i in range(nsource)]

        # Iterate through scanning time steps
        for _sidx in tqdm.tqdm_notebook(range(0,xyz_source.shape[0])):
            # Find all sources in scan window
            _in_fov1 = np.sort(self.tree_fov1.query_ball_point(xyz_source[_sidx].copy(order='C'), self.r_search))
            _in_fov2 = np.sort(self.tree_fov2.query_ball_point(xyz_source[_sidx].copy(order='C'), self.r_search))
            n_fov1 = len(_in_fov1)
            _in_fov = np.append(_in_fov1,_in_fov2).astype(int)
            if len(_in_fov)==0:
                continue

            # xyz coordinates of sources
            _xyz = xyz_source[_sidx]
            _uvw = np.einsum('ikj,j->ik',self._matrix[_in_fov],_xyz)

            _l = np.rad2deg(np.arctan2(_uvw[:,1],_uvw[:,0]))
            _b = np.rad2deg(np.arctan2(_uvw[:,2],np.sqrt(_uvw[:,0]**2.0+_uvw[:,1]**2.0)))

            # Test whether observations lie on an FoV.
            condition = (((np.abs(_l-self.l_fov_1)<1.0)&(_b<self.b_upp_1)&(_b>self.b_low_1))\
                               |((np.abs(_l-self.l_fov_2)<1.0)&(_b<self.b_upp_2)&(_b>self.b_low_2)))
            _in_fov = np.array(_in_fov)[condition]

            n_fov2 = np.sum(condition[n_fov1:])
            n_fov1 = np.sum(condition[:n_fov1])

            if n_fov1>0:
                condition_fov1 = self.tcb_at_gaia[_in_fov[:n_fov1]][1:] - self.tcb_at_gaia[_in_fov[:n_fov1]][:-1] > self.t_diff
                condition_fov1 = np.insert(condition_fov1, 0, True)

                _tidx_fov1 = np.vstack((_in_fov[:n_fov1][condition_fov1]-1, _in_fov[:n_fov1][condition_fov1]))
                _tidx_fov1.T[_tidx_fov1[0]<0] = np.array([0,1])
                tcbgaia_fov1 = self.linearbisect(_xyz, self.xyz_fov_1[_tidx_fov1], self.tcb_at_gaia[_tidx_fov1])

                # Apply scanning law gaps
                condition_gap1 = np.ones(np.sum(condition_fov1)).astype(bool)
                for ii in range(self._gaps.shape[0]):
                    condition_gap1 = condition_gap1&( (tcbgaia_fov1<self._gaps[ii,0])|(tcbgaia_fov1>self._gaps[ii,1]) )

                tgaia_fov1[_sidx] = list(tcbgaia_fov1[condition_gap1])
                nscan_fov1[_sidx] = np.sum(condition_gap1)

            if n_fov2>0:

                condition_fov2 = self.tcb_at_gaia[_in_fov[n_fov1:]][1:] - self.tcb_at_gaia[_in_fov[n_fov1:]][:-1] > self.t_diff
                condition_fov2 = np.insert(condition_fov2, 0, True)

                _tidx_fov2 = np.vstack((_in_fov[n_fov1:][condition_fov2]-1, _in_fov[n_fov1:][condition_fov2]))
                _tidx_fov2.T[_tidx_fov2[0]<0] = np.array([0,1])
                tcbgaia_fov2 = self.linearbisect(_xyz, self.xyz_fov_2[_tidx_fov2], self.tcb_at_gaia[_tidx_fov2])

                # # Apply scanning law gaps
                condition_gap2 = np.ones(np.sum(condition_fov2)).astype(bool)
                for ii in range(self._gaps.shape[0]):
                    condition_gap2 = condition_gap2&( (tcbgaia_fov2<self._gaps[ii,0])|(tcbgaia_fov2>self._gaps[ii,1]) )

                tgaia_fov2[_sidx] = list(tcbgaia_fov2[condition_gap2])
                nscan_fov2[_sidx] = np.sum(condition_gap2)

        return tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2

    #@ensure_flat_icrs
    def query(self, sources):
        """
        Returns the selection function at the requested coordinates.

        Args:
            sources (:obj:`astropy.coordinates.SkyCoord`): The coordinates to query.
                    (:obj:`scanninglaw.source.Source`): The coordinates to query.

        Returns:
            (:obj:`dict`): Observation times of each object by each FoV. Number of observations of each object by each FoV

        """

        if type(sources) == Source: coords = sources.coord.transform_to('icrs')
        else: coords = sources.transform_to('icrs')

        if type(coords.ra.deg)==np.ndarray:
            coord_shape = coords.ra.deg.shape
            radec_source = np.stack((coords.ra.deg.flatten(), coords.dec.deg.flatten())).T
        else:
            coord_shape = None
            radec_source = np.array([[coords.ra.deg],[coords.dec.deg]]).T

        ##### Compute tree
        xyz_source = np.stack([np.cos(np.deg2rad(radec_source[:,0]))*np.cos(np.deg2rad(radec_source[:,1])),
                               np.sin(np.deg2rad(radec_source[:,0]))*np.cos(np.deg2rad(radec_source[:,1])),
                               np.sin(np.deg2rad(radec_source[:,1]))]).T

        # Evaluate selection function
        if xyz_source.shape[0]>5e6:
              tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2 = self._scanning_law(xyz_source)
        else: tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2 = self._scanning_law_inverse(xyz_source)

        tgaia_fov1 = np.array(tgaia_fov1).reshape(coord_shape)
        tgaia_fov2 = np.array(tgaia_fov2).reshape(coord_shape)

        return tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2


def fetch(version='cogi_2020', fname=None):
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
        'dr2_nominal': {'filename': 'DEOPTSK-1327_Gaia_scanlaw.csv.gz'},
    }[version]

    # Download the data
    fetch_utils.dataverse_download_doi(
        doi,
        local_fname,
        file_requirements=requirements)
