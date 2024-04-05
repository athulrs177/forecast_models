from setuptools import setup, find_packages

setup(
    name='forecast_models',
    version='0.1.0',
    url="https://github.com/athulrs177/forecast_models.git",
    author="Athul Rasheeda Satheesh",
    author_email="athulrs177@gmail.com",
    packages=["forecast_models"],
    install_requires=[
        'numpy',
        'xarray',
        'pandas',
        'scikit-learn',
        'xgboost',
        'tensorflow-cpu',
        'git+https://github.com/evwalz/isodisreg.git',
        'scipy',
    ],
)
