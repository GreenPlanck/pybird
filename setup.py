from setuptools import setup

setup(
    name='pybird_w0wa',
    version='0.2.0',
    description='EFT predictions for biased tracers in redshift space.',
#    url='https://github.com/pierrexyz/pybird',
    author="Pierre Zhang and Guido D'Amico",
    license='MIT',
    packages=['pybird_w0wa'],
    install_requires=['numpy', 'scipy'],
    package_dir = {'pybird_w0wa': 'pybird_w0wa'},
    zip_safe=False,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python",
    ],
)
