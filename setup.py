from distutils.core import setup

setup(
    name='MothPriest',
    version='0.2',
    description='Module for creating intuitive and efficient file parsers',
    author='Cameron Churchwell',
    author_email='cameronchurchwell2024@u.northwestern.edu',
    install_requires=[
        'pathlib',
        'typing',
        'PIL'
    ],
    packages=['mothpriest'],
)