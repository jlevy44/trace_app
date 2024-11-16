# setup with TRACE entrypoint trace_app:main
from setuptools import setup

setup(
    name='TRACE',
    version='0.1',
    packages=['trace_app'],
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'trace=trace_app.trace_app:main',
        ],
    },
)
