import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

version = '0.1.dev1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'scikit-image'
    # 'Theano',  # we require a development version, see requirements.txt
    ]

extensions = [
    Extension(
        'bsds.correspond_pixels',
        [os.path.join('bsds', 'correspond_pixels.pyx')],
    ),
]

setup(
#     name = "py-bsds500",
#     version=version,
#     description="BSDS-500 access library and evaluation suite for Python",
#     long_description="\n\n".join([README]),
#     classifiers=[
#         "Development Status :: 1 - Alpha",
#         "Intended Audience :: Developers",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python :: 2.7",
#         # "Programming Language :: Python :: 3",
#         # "Programming Language :: Python :: 3.4",
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         ],
#     keywords="",
#     author="Geoffrey French",
#     # author_email="brittix1023 at gmail dot com",
#     url="https://github.com/Britefury/py-bsds500",
#     license="MIT",
#     # packages=find_packages(),
#     include_package_data=False,
#     zip_safe=False,
    install_requires=install_requires,

    packages = find_packages(),
    ext_modules = cythonize(extensions)
)
