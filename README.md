[![DOI](http://joss.theoj.org/papers/10.21105/joss.00695/status.svg)](https://doi.org/10.21105/joss.00695)

scanninglaw
==================

The ``scanninglaw`` package aims to provide and easy-to-use portal to the Gaia scanning law.
This package is entirely derivative of the truly excellent ``dustmaps`` package created by Gregory M. Green.
The ``scanninglaw`` package is a product of the [Completeness of the *Gaia*-verse (CoG)](https://www.gaiaverse.space/) collaboration.

Supported Scanning Laws
-----------------------------

The currently supported scanning laws are:

1. Gaia DR2 scanning law (cogi_2020, Boubert, Everall & Holl 2020, MNRAS)
2. Gaia DR2 scanning law (cog3_2020, Boubert, Everall, Fraser, Gration & Holl 2020)

To request addition of another sacnning law in this package, [file an issue on
GitHub](https://github.com/gaiaverse/scanninglaw/issues), or submit a pull request.


Installation
------------

Download the repository from [GitHub](https://github.com/gaiaverse/scanninglaw) and
then run:

    python setup.py install --large-data-dir=/path/where/you/want/large/data/files/stored

Alternatively, you can use the Python package manager `pip`:

    pip install scanninglaw


Getting the Data
----------------

To fetch the data for the GaiaDR2 scanning law, run:

    python setup.py fetch --map-name=cog3_2020

You can download the other scanning laws by changing "cog3_2020" to (other scanning laws will be added in future).

Alternatively, if you have used `pip` to install `scanninglaw`, then you can
configure the data directory and download the data by opening up a python
interpreter and running:

```python
from scanninglaw.config import config
config['data_dir'] = '/path/where/you/want/large/data/files/stored'

import scanninglaw.times
scanninglaw.times.fetch()
```

Querying the scanning law
-----------------

Scanning laws are queried using Source objects, which are a variant on the
[`astropy.coordinates.SkyCoord`](http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord)
object. This means that any coordinate system supported by `astropy` can be
used as input. For example, we can query the Gaia DR2 scanning law as follows:

```python
import scanninglaw.times as times
from scanninglaw.source import Source

dr2_sl = times.dr2_sl()

c = Source(
      '22h54m51.68s',
      '-51d11m44.19s',
      frame='icrs')
print(dr2_sl(c))
```


Above, we have used the ICRS coordinate system (the inputs are RA and Dec). We
can use other coordinate systems, such as Galactic coordinates, and we can
provide coordinate arrays. The following example uses both features:

```python
c = Source(
      [75.00000000, 130.00000000],
      [-89.00000000, 10.00000000],
      frame='galactic',
      unit='deg')
print(dr2_sl(c))
```


EDR3 Nominal scanning law
-------------------------

We've updated the repository for EDR3!

Fetch the nominal scanning law from the Gaia website:

```python
>>> from scanninglaw.config import config
>>> config['data_dir'] = '/path/where/you/want/large/data/files/stored'
>>>
>>> import scanninglaw.times
>>> scanninglaw.times.fetch(version='dr3_nominal')
```

And find when your star was observed:

```python
import scanninglaw.times as times
from scanninglaw.source import Source

dr3_sl = times.dr2_sl(version='dr3_nominal')

c = Source(
        '22h54m51.68s',
        '-51d11m44.19s',
        frame='icrs')
print(dr3_sl(c))
```

We haven't yet found the file for the DR3 published gaps but we'll incorporate those when we do!


Documentation
-------------

Read the full documentation at http://scanninglaw.readthedocs.io/en/latest/.


Citation
--------

If you make use of this software in a publication, please always cite
[Green (2018) in The Journal of Open Source Software](https://doi.org/10.21105/joss.00695).

You should also cite the papers behind the scanning laws you use.

1. cogi_2020 - Please cite Completeness of the Gaia-verse [Paper I](https://ui.adsabs.harvard.edu/abs/2020arXiv200414433B/abstract).
2. cog3_2020 - Please cite Completeness of the Gaia-verse [Paper III](https://ui.adsabs.harvard.edu/abs/2020arXiv201110578B/abstract).

Development
-----------

Development of `scanninglaw` takes place on GitHub, at
https://github.com/gaiaverse/scanninglaw. Any bugs, feature requests, pull requests,
or other issues can be filed there. Contributions to the software are welcome.
