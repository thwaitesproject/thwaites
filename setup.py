#!/usr/bin/env python

from distutils.core import setup
from glob import glob

import versioneer

cmdclass = versioneer.get_cmdclass()

setup(name='thwaites',
      cmdclass=cmdclass,
      version=versioneer.get_version(),
      description='Finite element ocean model',
      author='Stephan Kramer',
      author_email='s.kramer@imperial.ac.uk',
      url='https://github.com/thwaites/thwaites',
      packages=['thwaites'],
     )
