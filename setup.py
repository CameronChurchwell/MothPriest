from distutils.core import setup

setup(
    name='MothPriest',
    version='0.3',
    description='Module for creating intuitive and efficient file parsers',
    author='Cameron Churchwell',
    author_email='cameronchurchwell2024@u.northwestern.edu',
    install_requires=[
        'pathlib',
        'typing',
        'Pillow'
    ],
    packages=['mothpriest'],
)