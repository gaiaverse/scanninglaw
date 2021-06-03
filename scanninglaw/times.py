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

version_colnames = {
    'cogi_2020':  ['JulianDayNumberRefEpoch2010TCB@Gaia', 'JulianDayNumberRefEpoch2010TCB@Barycentre_1', 'JulianDayNumberRefEpoch2010TCB@Barycentre_2',
                'ra_FOV_1(deg)', 'dec_FOV_1(deg)', 'scanPositionAngle_FOV_1(deg)', 'ra_FOV_2(deg)', 'dec_FOV_2(deg)', 'scanPositionAngle_FOV_2(deg)'],
    'cog3_2020':  ['JulianDayNumberRefEpoch2010TCB@Gaia', 'JulianDayNumberRefEpoch2010TCB@Barycentre_1', 'JulianDayNumberRefEpoch2010TCB@Barycentre_2',
                'ra_FOV_1(deg)', 'dec_FOV_1(deg)', 'scanPositionAngle_FOV_1(deg)', 'ra_FOV_2(deg)', 'dec_FOV_2(deg)', 'scanPositionAngle_FOV_2(deg)'],
    'dr2_nominal':  ['JulianDayNumberRefEpoch2010TCB@Gaia', 'JulianDayNumberRefEpoch2010TCB@Barycentre_1', 'JulianDayNumberRefEpoch2010TCB@Barycentre_2',
                'ra_FOV_1(deg)', 'dec_FOV_1(deg)', 'scanPositionAngle_FOV_1(deg)', 'ra_FOV_2(deg)', 'dec_FOV_2(deg)', 'scanPositionAngle_FOV_2(deg)'],
    'dr3_nominal': ['jd_time', 'bjd_fov1', 'bjd_fov2',
                'ra_fov1', 'dec_fov1', 'scan_angle_fov1', 'ra_fov2', 'dec_fov2', 'scan_angle_fov2']
    }

version_filenames = {
    'cogi_2020': 'cog_dr2_scanning_law_v1.csv',
    'cog3_2020': 'cog_dr2_scanning_law_v2.csv',
    'dr2_nominal': 'DEOPTSK-1327_Gaia_scanlaw.csv',
    'dr3_nominal': 'CommandedScanLaw_001.csv'
    }

version_trange = {
    'cogi_2020': [1192.13,3750.56],
    'cog3_2020': [1192.13,3750.56],
    'dr2_nominal': [1192.13,3750.56],
    'dr3_nominal': [1192.13,5230.09]
    }



class Times(ScanningLaw):
    """
    Queries the Gaia DR2 selection function (Boubert & Everall, 2019).
    """

    def __init__(self, map_fname=None,
                        version='cog3_2020', sample='Astrometry',
                        load_fractions=False,
                        fractions='cog_dr2_gaps_and_fractions_v1.h5',
                        ephemeris='horizons_results_gaia.txt',
                        require_persistent=False, test=False):
        """
        Args:
            map_fname (Optional[:obj:`str`]): Filename of the Boubert,Everall,Holl 2020 scanning law. Defaults to
                :obj:`None`, meaning that the default location is used.
            version (Optional[:obj:`str`]): The scanning law version to download. Valid versions
                are :obj:`'cogi_2020'`
                are :obj:`'cog3_2020'`
                are :obj:`'dr2_nominal'`
                Defaults to :obj:`'cog3_2020'`.
            sample (Optional[:obj:`str`]): The Gaia data sample being used. Valid options are
                are :obj:`'Astrometry'`
                are :obj:`'Photometry'`
                are :obj:`'Spectroscopy'`
                Defaults to :obj:`'Astrometry'`.
            fractions (Optional[:obj:`str`]): File containing fractions from CoG III.
                Defaults to :obj:`'cog_dr2_gaps_and_fractions_v1.h5'`.
            ephemeris (Optional[:obj:`str`]): File containing Gaia ephemeris data.
                Defaults to :obj:`'horizons_results_gaia.txt'`.
        """

        if version=='cog': version='cog3_2020'

        if version=='dr3_nominal': sample+='_dr3'

        if map_fname is None:
            map_fname = os.path.join(data_dir(), 'cog', '{}'.format(version_filenames[version]))
        self.fractions_fname = os.path.join(data_dir(), 'cog', fractions)
        local_dirname = os.path.dirname(__file__)
        gaps_fname = os.path.join(local_dirname, 'data', '{}.csv'.format(sample))
        ephemeris_fname = os.path.join(local_dirname, 'data', ephemeris)

        if version=='cogi_2020': self.use_aberration=False
        else: self.use_aberration=True

        t_start = time()

        # Load auxilliary data
        print('Loading auxilliary data ...')

        ## Load in scanning law
        _columns = version_colnames[version]
        if test: _data = pd.read_csv(map_fname, usecols=_columns, nrows=1000000)
        else: _data = pd.read_csv(map_fname, usecols=_columns)
        _keys = ['tcb_at_gaia','tcb_at_bary1','tcb_at_bary2','ra_fov_1','dec_fov_1','angle_fov_1','ra_fov_2','dec_fov_2','angle_fov_2']
        _box = {}
        for j,k in zip(_columns,_keys): _box[k] = _data[j].values
        _box['scan_idx'] = np.arange(len(_box['ra_fov_1']))
        order = np.argsort(_box['tcb_at_gaia'])
        for k in _box.keys(): _box[k] = _box[k][order]

        # Interpolate scan angles
        self.angle_interp = [scipy.interpolate.interp1d(_box['tcb_at_gaia'], _box['angle_fov_1'],
                                                   bounds_error=False, fill_value=0.),\
                            scipy.interpolate.interp1d(_box['tcb_at_gaia'], _box['angle_fov_2'],
                                                   bounds_error=False, fill_value=0.)]

        ## Load gaps
        if version=='dr3_nominal':
            _columns = ['tbeg', 'tend'];
            _data = pd.read_csv(gaps_fname, usecols=_columns)
            self._gaps = obmt2tcbgaia(np.vstack((_data['tbeg'].values, _data['tend'].values)).T)
            if sample=='Astrometry_dr3':
                self._gaps = np.vstack((np.array([-np.inf, obmt2tcbgaia(version_trange[version][0])])[np.newaxis,:], self._gaps ))
                self._gaps = np.vstack(( self._gaps, np.array([obmt2tcbgaia(version_trange[version][1]), np.inf])[np.newaxis,:],  ))
        else:
            _columns = ['start [rev]', 'end [rev]', 'persistent'];
            _data = pd.read_csv(gaps_fname, usecols=_columns)
            self._gaps = obmt2tcbgaia(np.vstack((_data['start [rev]'].values, _data['end [rev]'].values)).T)
            if require_persistent: self._gaps=self._gaps[_data['persistent']==True]
            if sample=='Astrometry':
                self._gaps = np.vstack((np.array([-np.inf, obmt2tcbgaia(version_trange[version][0])])[np.newaxis,:], self._gaps ))
                self._gaps = np.vstack(( self._gaps, np.array([obmt2tcbgaia(version_trange[version][1]), np.inf])[np.newaxis,:],  ))

        ## Load fraction interpolations
        if load_fractions: self.load_fractions()

        ## Load Gaia ephemeris data
        # Define units
        speed_of_light_AU_per_day = 299792458.0*(86400.0/149597870.700/1e3)
        # Prepare ephem data
        gaia_ephem_data = pd.read_csv(ephemeris_fname,skiprows=64)
        gaia_ephem_box = {k:gaia_ephem_data[k].values for k in ['JDTDB','X','Y','Z','VX','VY','VZ']}
        self.gaia_ephem_velocity = interpolate.interp1d(gaia_ephem_box['JDTDB']-2455197.5,np.stack([gaia_ephem_box['VX'],gaia_ephem_box['VY'],gaia_ephem_box['VZ']])/speed_of_light_AU_per_day,kind='cubic')

        t_auxilliary = time()

        # Compute 3D spherical projections
        self.xyz_fov_1 = np.stack([np.cos(np.deg2rad(_box['ra_fov_1']))*np.cos(np.deg2rad(_box['dec_fov_1'])),
                              np.sin(np.deg2rad(_box['ra_fov_1']))*np.cos(np.deg2rad(_box['dec_fov_1'])),
                              np.sin(np.deg2rad(_box['dec_fov_1']))]).T
        self.xyz_fov_2 = np.stack([np.cos(np.deg2rad(_box['ra_fov_2']))*np.cos(np.deg2rad(_box['dec_fov_2'])),
                              np.sin(np.deg2rad(_box['ra_fov_2']))*np.cos(np.deg2rad(_box['dec_fov_2'])),
                              np.sin(np.deg2rad(_box['dec_fov_2']))]).T
        ## Compute rotation matrices
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
        # CCD width: 0.7deg, Add aberration freedom: maximum 21arcsec
        self.r_search = np.tan(np.deg2rad(0.35*np.sqrt(2) + 25./3600))
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

        self.magbins = np.array([5, 13,  16, 16.3, 17, 17.2, 18, 18.1, 19, 19.05, 19.95,
                                 20, 20.3, 20.4, 20.5, 20.6, 20.7,20.8,20.9, 21])

        t_interpolator = time()

        t_finish = time()

        print('t = {:.3f} s'.format(t_finish - t_start))
        print('  auxilliary: {: >7.3f} s'.format(t_auxilliary-t_start))
        print('          sf: {: >7.3f} s'.format(t_sf-t_auxilliary))
        print('interpolator: {: >7.3f} s'.format(t_interpolator-t_sf))

    def load_fractions(self):
        ##### Load fraction
        print('Loading and Interpolating fractions, this only needs to be done once for the class instance.')
        print('Loading...')
        with h5py.File(self.fractions_fname, "r") as f:
            scanninglaw_times = f['times'][:]
            scanninglaw_gaps = f['gaps'][:]
            scanninglaw_fractions = f['fractions'][:]
        probability_time_series = scanninglaw_fractions*scanninglaw_gaps[np.newaxis,:]
        print('Interpolating...')
        #gap_interp = scipy.interpolate.interp1d(scanninglaw_times, scanninglaw_gaps, kind='nearest', bounds_error=False, fill_value='extrapolate')
        self.probability_mag_interp = [scipy.interpolate.interp1d(scanninglaw_times, probability_time_series[j], kind='nearest', bounds_error=False, fill_value='extrapolate')
                                    for j in range(probability_time_series.shape[0])]

    def linearbisect(self, xyz_obj, xyz_line, tgaia_line):

        _d_line = xyz_line[1]-xyz_line[0]
        _d_obj = xyz_obj-xyz_line[0]
        _line_sqlen = np.sum((_d_line)**2, axis=1)

        _d_tobs = (tgaia_line[1]-tgaia_line[0]) * np.sum(_d_line * _d_obj, axis=1) / _line_sqlen

        return tgaia_line[0]+_d_tobs

    def aberration(self, xyz_source, _t_now):

        # Aberration correction
        _gaia_velocity = self.gaia_ephem_velocity(_t_now)

        if len(xyz_source.shape)==1:
              _xyz = xyz_source + _gaia_velocity.T
        else: _xyz = xyz_source + _gaia_velocity
        _xyz = _xyz/np.linalg.norm(_xyz,axis=1)[:,np.newaxis]

        return _xyz

    def _scanning_law(self, xyz_source):

        # Run this code for large numbers of sources (5 million +)

        nsource = xyz_source.shape[0]
        t_previous = -99999*np.ones(nsource)
        t_box = {i:[] for i in range(nsource)}
        idx_box = {i:[] for i in range(nsource)}

        tgaia_fov1 = [[] for i in range(nsource)]
        tgaia_fov2 = [[] for i in range(nsource)]
        b_fov1 = [[] for i in range(nsource)]
        b_fov2 = [[] for i in range(nsource)]
        nscan_fov1 = [0 for i in range(nsource)]
        nscan_fov2 = [0 for i in range(nsource)]

        tree_source = spatial.cKDTree(xyz_source)

        # Iterate through scanning time steps
        for _tidx in self.tqdm_foo(range(0,self.xyz_fov_1.shape[0]), disable=not self.progressbar):
            _t_now = self.tcb_at_gaia[_tidx]
            # Find all sources in scan window
            _in_fov = tree_source.query_ball_point([self.xyz_fov_1[_tidx].copy(order='C'),
                                                    self.xyz_fov_2[_tidx].copy(order='C')],self.r_search)
            n_fov1 = len(_in_fov[0])
            _in_fov = _in_fov[0]+_in_fov[1]
            if len(_in_fov) == 0:
                continue

            # xyz coordinates of sources corrected for aberration if not cogi_2020
            if self.use_aberration:_xyz = self.aberration(xyz_source[_in_fov], _t_now)
            else: _xyz = xyz_source[_in_fov]
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
                    angle_fov1[_sidx].append(self.angle_interp[0](tcbgaia_fov1[ii]))
                    b_fov1[_sidx].append(_b[_where[ii]])
                    nscan_fov1[_sidx] += 1
                    #print(self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2])
                for ii in range(len(_valid)-n_fov1):
                    _sidx = _valid[ii+n_fov1]
                    tgaia_fov2[_sidx].append(tcbgaia_fov2[ii])
                    angle_fov2[_sidx].append(self.angle_interp[1](tcbgaia_fov2[ii]))
                    b_fov2[_sidx].append(_b[_where[ii]])
                    nscan_fov2[_sidx] += 1
                    #print(self.tcb_at_gaia[max(0,_tidx-1):_tidx+2][:2])

        return tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2, b_fov1, b_fov2

    def _scanning_law_inverse(self, xyz_source):

        # Run this code for small numbers of sources (less than 5 million)

        nsource = xyz_source.shape[0]
        t_previous = -99999*np.ones(nsource)
        t_box = {i:[] for i in range(nsource)}
        idx_box = {i:[] for i in range(nsource)}

        tgaia_fov1 = [[] for i in range(nsource)]
        tgaia_fov2 = [[] for i in range(nsource)]
        angle_fov1 = [[] for i in range(nsource)]
        angle_fov2 = [[] for i in range(nsource)]
        b_fov1 = [[] for i in range(nsource)]
        b_fov2 = [[] for i in range(nsource)]
        nscan_fov1 = [0 for i in range(nsource)]
        nscan_fov2 = [0 for i in range(nsource)]

        # Iterate through scanning time steps
        for _sidx in self.tqdm_foo(range(0,xyz_source.shape[0]), disable=not self.progressbar):
            # Find all sources in scan window
            _in_fov1 = np.sort(self.tree_fov1.query_ball_point(xyz_source[_sidx].copy(order='C'), self.r_search))
            _in_fov2 = np.sort(self.tree_fov2.query_ball_point(xyz_source[_sidx].copy(order='C'), self.r_search))
            n_fov1 = len(_in_fov1)
            _in_fov = np.append(_in_fov1,_in_fov2).astype(int)
            if len(_in_fov)==0:
                continue

            # xyz coordinates of sources corrected for aberration
            _t_now = self.tcb_at_gaia[_in_fov]
            if self.use_aberration: _xyz = self.aberration(xyz_source[_sidx], _t_now)
            else: _xyz = np.repeat([xyz_source[_sidx],], len(_in_fov), axis=0)
            _uvw = np.einsum('ikj,ij->ik',self._matrix[_in_fov],_xyz)

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
                tcbgaia_fov1 = self.linearbisect(_xyz[condition][:n_fov1][condition_fov1], self.xyz_fov_1[_tidx_fov1], self.tcb_at_gaia[_tidx_fov1])

                # Apply scanning law gaps
                condition_gap1 = np.ones(np.sum(condition_fov1)).astype(bool)
                for ii in range(self._gaps.shape[0]):
                    condition_gap1 = condition_gap1&( (tcbgaia_fov1<self._gaps[ii,0])|(tcbgaia_fov1>self._gaps[ii,1]) )

                tgaia_fov1[_sidx] = list(tcbgaia_fov1[condition_gap1])
                b_fov1[_sidx] = list(_b[condition][:n_fov1][condition_fov1][condition_gap1])
                nscan_fov1[_sidx] = np.sum(condition_gap1)
                angle_fov1[_sidx] = list(self.angle_interp[0](tcbgaia_fov1[condition_gap1]))

            if n_fov2>0:

                condition_fov2 = self.tcb_at_gaia[_in_fov[n_fov1:]][1:] - self.tcb_at_gaia[_in_fov[n_fov1:]][:-1] > self.t_diff
                condition_fov2 = np.insert(condition_fov2, 0, True)

                _tidx_fov2 = np.vstack((_in_fov[n_fov1:][condition_fov2]-1, _in_fov[n_fov1:][condition_fov2]))
                _tidx_fov2.T[_tidx_fov2[0]<0] = np.array([0,1])
                tcbgaia_fov2 = self.linearbisect(_xyz[condition][n_fov1:][condition_fov2], self.xyz_fov_2[_tidx_fov2], self.tcb_at_gaia[_tidx_fov2])

                # # Apply scanning law gaps
                condition_gap2 = np.ones(np.sum(condition_fov2)).astype(bool)
                for ii in range(self._gaps.shape[0]):
                    condition_gap2 = condition_gap2&( (tcbgaia_fov2<self._gaps[ii,0])|(tcbgaia_fov2>self._gaps[ii,1]) )

                tgaia_fov2[_sidx] = list(tcbgaia_fov2[condition_gap2])
                b_fov2[_sidx] = list(_b[condition][n_fov1:][condition_fov2][condition_gap2])
                nscan_fov2[_sidx] = np.sum(condition_gap2)
                angle_fov2[_sidx] = list(self.angle_interp[1](tcbgaia_fov2[condition_gap2]))

        return tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2, b_fov1, b_fov2, angle_fov1, angle_fov2

    def _get_magidx(self, G):

        """
        Returns the magnitude bin ids for the given magnitudes.

        Args:
            G (:obj:`np.ndarray`): G magnitude.

        Returns:
            magidx (:obj:`dict`): bin ID of magnitude

        """

        magidx = np.zeros(G.shape).astype(int) - 1

        for mag in self.magbins:
            magidx += (G>mag).astype(int)

        magidx[magidx==len(self.magbins)-1] = -99
        magidx[magidx==-1] = -99

        return magidx

    def _scanning_fraction(self, magidx, tgaia_fov1, tgaia_fov2):

        """
        Returns the scan fractions.

        Args:
            magidx (:obj:`np.ndarray`): magnitude bin ids.
            tgaia_fov1 (:obj:`np.array`): observation times in FoV1.
            tgaia_fov2 (:obj:`np.array`): observation times in FoV2.

        Returns:
            fraction_fov1 (:obj:`np.array`): scan fraction in FoV1.
            fraction_fov2 (:obj:`np.array`): scan fraction in FoV2.
        """

        fraction_fov1 = [[] for i in range(len(magidx))]
        fraction_fov2 = [[] for i in range(len(magidx))]

        # Iterate through scanning time steps
        for _sidx in self.tqdm_foo(range(0,len(magidx)), disable=not self.progressbar):

            if magidx[_sidx]==-99:
                fraction_fov1[_sidx] = [np.nan for i in range(len(tgaia_fov1[_sidx]))]
                fraction_fov2[_sidx] = [np.nan for i in range(len(tgaia_fov2[_sidx]))]
                continue

            fraction_fov1[_sidx] = self.probability_mag_interp[magidx[_sidx]](tgaia_fov1[_sidx])
            fraction_fov2[_sidx] = self.probability_mag_interp[magidx[_sidx]](tgaia_fov2[_sidx])

        return fraction_fov1, fraction_fov2

    #@ensure_flat_icrs
    def query(self, sources, return_times=True, return_counts=True, return_fractions=False, return_angles=False,
                             return_acoffset=False, fov=12, progress=False):
        """
        Returns the scanning law at the requested coordinates.

        Args:
            sources (:obj:`astropy.coordinates.SkyCoord`): The coordinates to query.
                    (:obj:`scanninglaw.source.Source`): The coordinates to query.

        KwArgs:
            return_counts (:obj:`bool`)
            return_fractions (:obj:`bool`)
            fov (:obj:`int`). Which fov to return 1, 2 or 12 for both FoVs
            progress (:obj:`bool` or `str`). False - No progress bar. True - tqdm.tqdm progressbar. 'notebook' - tqdm.tqdm_notebook progress bar (for Jupyter notebooks)

        Returns:
            (:obj:`dict`): Observation times of each object by each FoV. Number of observations of each object by each FoV

        """

        if not fov in (1,2,12): raise ValueError('Invalid value for kwarg fov. fov must be 1, 2 or 12.')

        if progress=='notebook':
            self.tqdm_foo = tqdm.tqdm_notebook
            self.progressbar=True
        else:
            self.tqdm_foo = tqdm.tqdm
            self.progressbar=bool(progress)

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
              tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2, \
              b_fov1, b_fov2, angle_fov1, angle_fov2 = self._scanning_law(xyz_source)
        else: tgaia_fov1, tgaia_fov2, nscan_fov1, nscan_fov2, \
              b_fov1, b_fov2, angle_fov1, angle_fov2 = self._scanning_law_inverse(xyz_source)

        # Extract Gaia G magnitude
        try: G = sources.photometry.measurement['gaia_g']; G_given = True
        except AttributeError: G_given=False

        if (not G_given)&(return_fractions):
            raise ValueError("return_fractions=true but gaia_g not given. gaia_g must be set in Source instance to return fractions. \ne.g. Source('12h30m25.3s', '15d15m58.1s', frame='icrs', photometry={'gaia_g':21.0})")
        elif (G_given)&(return_fractions):
            if type(G)==np.ndarray: G = G.flatten()
            else: G = np.array([G])

            if not hasattr(self, 'probability_mag_interp'): self.load_fractions()

            Gidx = self._get_magidx(G)
            fraction_fov1, fraction_fov2 = self._scanning_fraction(Gidx, tgaia_fov1, tgaia_fov2)

            # tgaia_fov1 = np.array(tgaia_fov1).reshape(coord_shape)
            # tgaia_fov2 = np.array(tgaia_fov2).reshape(coord_shape)
            # b_fov1 = np.array(b_fov1).reshape(coord_shape)
            # b_fov2 = np.array(b_fov2).reshape(coord_shape)
            fraction_fov1 = np.array(fraction_fov1,dtype=object).reshape(coord_shape)
            fraction_fov2 = np.array(fraction_fov2,dtype=object).reshape(coord_shape)

        tgaia_fov1 = np.array(tgaia_fov1,dtype=object).reshape(coord_shape)
        tgaia_fov2 = np.array(tgaia_fov2,dtype=object).reshape(coord_shape)
        b_fov1 = np.array(b_fov1,dtype=object).reshape(coord_shape)
        b_fov2 = np.array(b_fov2,dtype=object).reshape(coord_shape)
        angle_fov1 = np.array(angle_fov1,dtype=object).reshape(coord_shape)
        angle_fov2 = np.array(angle_fov2,dtype=object).reshape(coord_shape)
        nscan_fov1 = np.array(nscan_fov1,dtype=object).reshape(coord_shape)
        nscan_fov2 = np.array(nscan_fov2,dtype=object).reshape(coord_shape)

        ret = {}
        if return_times: ret['times']=[]
        if return_angles: ret['angles']=[]
        if return_counts: ret['counts']=[]
        if return_fractions: ret['fractions']=[]
        if return_acoffset: ret['acoffset']=[]
        if (fov in (1,12)):
            if return_times: ret['times'] += [tgaia_fov1,]
            if return_angles: ret['angles'] += [angle_fov1,]
            if return_counts: ret['counts'] += [nscan_fov1,]
            if return_fractions: ret['fractions'] += [fraction_fov1,]
            if return_acoffset: ret['acoffset'] += [b_fov1,]
        if (fov in (2,12)):
            if return_times: ret['times'] += [tgaia_fov2,]
            if return_angles: ret['angles'] += [angle_fov2,]
            if return_counts: ret['counts'] += [nscan_fov2,]
            if return_fractions: ret['fractions'] += [fraction_fov2,]
            if return_acoffset: ret['acoffset'] += [b_fov2,]

        return ret


def fetch(version='cog3_2020', fname=None):
    """
    Downloads the specified version of the Gaia DR2 scanning law.

    Args:
        version (Optional[:obj:`str`]): The map version to download. Valid versions are
            :obj:`'cogi_2020'` (Boubert, Everall & Holl 2020)
            :obj:`'cog3_2020'` (Boubert, Everall, Fraser, Gration & Holl 2020)
            :obj:`'dr2_nominal'` (Prusti+ 2016, Brown+ 2018)
            Defaults to :obj:`'cog3_2020'`.

    Raises:
        :obj:`ValueError`: The requested version of the map does not exist.

        :obj:`DownloadError`: Either no matching file was found under the given DOI, or
            the MD5 sum of the file was not as expected.

        :obj:`requests.exceptions.HTTPError`: The given DOI does not exist, or there
            was a problem connecting to the Dataverse.
    """

    if not version in version_filenames:
        raise ValueError('{0} not valid. Valid map names are: {1}'.format(version, list(version_filenames.keys())))

    requirements = {
        'cogi_2020': {'filename': 'cog_dr2_scanning_law_v1.csv.gz'},
        'cog3_2020': {'filename': 'cog_dr2_scanning_law_v2.csv'},
        'dr2_nominal': {'filename': 'DEOPTSK-1327_Gaia_scanlaw.csv.gz'},
        'dr3_nominal': {'filename': 'CommandedScanLaw_001.csv.gz'},
    }[version]
    #if version=='cog3_2020': local_fname = os.path.join(data_dir(), 'cog', '{}.csv'.format(version))
    #else: local_fname = os.path.join(data_dir(), 'cog', '{}.csv.gz'.format(version))
    local_fname = os.path.join(data_dir(), 'cog', requirements['filename'])

    if not (version=='dr3_nominal'):
        # Download gaps and fractions
        fetch_utils.dataverse_download_doi(
            '10.7910/DVN/ST8TSM',
            os.path.join(data_dir(), 'cog', 'cog_dr2_gaps_and_fractions_v1.h5'),
            file_requirements={'filename': 'cog_dr2_gaps_and_fractions_v1.h5'})

    if (version=='dr2_nominal')&(fname is None):
        raise ValueError("\nNominal scanning law at ftp.cosmos.esa.int/GAIA_PUBLIC_DATA/GaiaScanningLaw/DEOPTSK-1327_Gaia_scanlaw.csv.gz.\n"\
                          "Download .gz file using ftp client then run fetch(version='dr2_nominal', fname='path/to/file'). ")
    elif version=='dr2_nominal':
        fetch_utils.move_file_location(fname, local_fname)
        return None


    doi = {
        'cogi_2020': '10.7910/DVN/OFRA78',
        'cog3_2020': '10.7910/DVN/MYIPLH',
        'dr3_nominal': 'http://cdn.gea.esac.esa.int/Gaia/gedr3/auxiliary/commanded_scan_law/CommandedScanLaw_001.csv.gz'
    }
    # Raise an error if the specified version of the map does not exist
    try:
        doi = doi[version]
    except KeyError as err:
        raise ValueError('Version "{}" does not exist. Valid versions are: {}'.format(
            version,
            ', '.join(['"{}"'.format(k) for k in doi.keys()])
        ))

    if version.endswith('nominal'):
        # Download the data
        fetch_utils.download(doi, fname=local_fname) #file_requirements=requirements)
    else:
        # Download the data
        fetch_utils.dataverse_download_doi(
            doi,
            local_fname,
            file_requirements=requirements)
