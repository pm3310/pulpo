from setuptools import find_packages, setup
import sys

CORE_REQUIREMENTS = [
        'numpy>=1.18.0, <1.18.99',
        'six>=1.14, <1.14.99',
        'future>=0.18.0, <0.18.99'
]

if sys.version_info < (3, 7):
    REQUIRES = CORE_REQUIREMENTS + ["dataclasses"]
else:
    REQUIRES = CORE_REQUIREMENTS

with open('README.md') as f:
    long_description = f.read()

setup(
    name='pulpo',
    version='0.0.1',
    setup_cfg=True,
    python_requires='~=3.6',
    packages=find_packages(where='.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['setuptools>=39.1.0'],
    url='https://github.com/pm3310/pulpo',
    install_requires=REQUIRES,
    test_suite='tests',
    zip_safe=True
)
