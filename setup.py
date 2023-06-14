#!/usr/bin/env python
import re
import os
import sys
import setuptools


# warn against python 2
if sys.version_info[0] == 2:
    print('tensiometer does not support Python 2, \
           please upgrade to Python 3')
    sys.exit(1)


# version control:
def find_version():
    version_file = open(os.path.join(os.path.dirname(__file__),
                                     'nf/__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


# long description (parse readme):
def get_long_description():
    with open('README.rst',  encoding='utf-8-sig') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return ''.join(lines[:i])


# get requirements:
def get_requirements():
    requirements = []
    with open('requirements.txt',  encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if '>=' in line:
                pc, ver = line.split('>=')
                requirements.append(pc+' (>='+ver+')')
            else:
                requirements.append(line)
    return requirements


# setup:
setuptools.setup(name='CAMELS_NPE',
                 description='NPE for CAMELS',
                 author='Shiam Pandey',
                 license='GPL',
                 project_urls={
                    'Source': 'https://github.com/shivampcosmo/CAMLES_NPE',
                    'Tracker': 'https://github.com/shivampcosmo/CAMLES_NPE/issues',
                    },
                 packages=setuptools.find_packages(),
                 platforms='any',
                 install_requires=get_requirements(),
                 classifiers=[
                    'Development Status :: 4 - Beta',
                    'Operating System :: OS Independent',
                    'Intended Audience :: Science/Research',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    ],
                 python_requires='>=3.7',
                 zip_safe=False
                 )