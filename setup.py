from setuptools import setup, find_packages

setup(
    name="etl_logic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'geopandas',
        'polars',
        'requests',
        'tqdm',
        'pyarrow',
        'shapely',
        'openpyxl'
    ],
) 