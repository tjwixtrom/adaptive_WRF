from setuptools import setup

setup(
    name='analogue_algorithm',
    version='0.1',
    packages=['analogue_algorithm'],
    requires=['netcdf4', 'numpy', 'scipy', 'dask', 'xarray', 'pandas', 'pyresample'],
    url='',
    license='BSD-3',
    author='Tyler Wixtrom',
    author_email='tyler.wixtrom@ttu.edu',
    description='Code for analogue calculation'
)
