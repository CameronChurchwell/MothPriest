from distutils.core import setup

setup(
    name='MothPriest',
    version='1.0.0',
    description='Module for creating intuitive and efficient file parsers',
    author='Cameron Churchwell',
    author_email='cameronchurchwell@icloud.com',
    install_requires=[
        'pathlib',
        'typing',
        'Pillow',
        'pytest'
    ],
    packages=['mothpriest'],
)